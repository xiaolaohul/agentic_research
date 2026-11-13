"""Serve the existing BrowseComp-Plus reranker over a loopback SSH tunnel."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
import json
from math import isfinite
from time import perf_counter
from types import SimpleNamespace

from src.reranker import Reranker
from src.schemas import Hit
from src.utils import ROOT, load_config, path, setup_environment


MAX_REQUEST_BYTES = 32 * 1024 * 1024


def load_reranker():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
    cfg = load_config(ROOT / 'configs' / 'default.yaml')
    setup_environment(cfg)
    config = dict(cfg['reranker'], device='cuda', depth=100, max_length=512)
    repo = config['model'].replace('/', '--')
    cache = path(cfg['storage']['model_cache']) / 'hub'
    snapshot = cache / f"models--{repo}" / 'snapshots' / config['revision']
    if not (snapshot / 'model.safetensors').is_file():
        from huggingface_hub import snapshot_download

        print(f"Downloading {config['model']} at {config['revision']} ...", flush=True)
        snapshot = snapshot_download(
            config['model'], revision=config['revision'], cache_dir=cache,
            allow_patterns=['*.json', 'model.safetensors', 'sentencepiece.bpe.model'])
    reranker = Reranker([], dict(config, model=str(snapshot)))
    software = {'torch': version('torch'), 'transformers': version('transformers'),
                'gpu': torch.cuda.get_device_name(0)}
    return reranker, config, software


def make_handler(reranker, config, software):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                self.send_error(404)
                return
            self.send_json(200, {'config': config, 'software': software})

        def do_POST(self):
            if self.path != '/rerank':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError('Invalid request size.')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object.')
                query, rows = payload.get('query'), payload.get('documents')
                if (not isinstance(query, str) or not isinstance(rows, list)
                        or not 1 <= len(rows) <= config['depth']):
                    raise ValueError('Expected a query and a nonempty list of candidate documents.')
                docs = {}
                for row in rows:
                    if (not isinstance(row, list) or len(row) != 2
                            or not all(isinstance(value, str) for value in row)
                            or not row[0] or row[0] in docs):
                        raise ValueError('Expected unique document IDs and text.')
                    doc_id, text = row
                    docs[doc_id] = SimpleNamespace(search_text=text)
            except (ValueError, UnicodeError) as exc:
                self.send_json(400, {'error': str(exc)})
                return

            # HTTPServer handles one request at a time. Keep the weights loaded and
            # release each request's document texts when scoring completes.
            hits = [Hit(doc_id, rank, 0.0, rank, 0.0) for rank, doc_id in enumerate(docs, 1)]
            reranker.docs = docs
            started = perf_counter()
            try:
                ranked = reranker.rerank(query, hits)
                model_seconds = perf_counter() - started
                if len(ranked) != len(hits) or any(not isfinite(hit.score) for hit in ranked):
                    raise ValueError('Model returned incomplete or non-finite scores.')
            except (RuntimeError, ValueError) as exc:
                self.log_error('Reranking failed: %s', exc)
                self.send_json(500, {'error': 'Reranking failed. See the AutoDL terminal.'})
                return
            finally:
                reranker.docs = {}
            self.send_json(200, {'config': config, 'software': software, 'model_seconds': model_seconds,
                                 'ranking': [[hit.doc_id, hit.score] for hit in ranked]})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()
    reranker, config, software = load_reranker()
    with HTTPServer(('127.0.0.1', args.port), make_handler(reranker, config, software)) as server:
        print(f'Reranker ready at http://127.0.0.1:{server.server_port} on {software["gpu"]}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
