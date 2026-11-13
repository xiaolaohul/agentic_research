"""Replay saved BrowseComp-Plus rankings with the project's existing reranker."""
from dataclasses import replace
from hashlib import sha256
from http.client import HTTPConnection, HTTPException
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from .dataset import load_candidate_documents
from .evaluation import score_rankings
from ..schemas import Hit, Query
from ..utils import ROOT, digest, load_config, path, read_json


def load_saved_candidates(source: Path, queries: list[Query], revisions: dict[str, str],
                          depth: int, method: str, *, require_full: bool):
    metadata = read_json(source / 'run.json')
    source_revisions = metadata['download_revisions']
    if (metadata['method'] != method
            or source_revisions['benchmark'] != revisions['benchmark']
            or metadata['max_hits_per_query'] != depth):
        raise ValueError(f'Expected a saved BrowseComp-Plus {method} run at the same question revision and depth.')
    if method == 'bge' and source_revisions.get('corpus') != revisions['corpus']:
        raise ValueError('The saved BGE run used a different corpus revision.')
    document_count = metadata['index_document_count']
    if (method == 'bm25' or require_full) and (metadata['query_count'] != 830 or document_count != 100_195):
        raise ValueError(f'Expected a full {method} baseline with 830 questions and 100,195 documents.')
    run_path, queries_path = source / f'{method}.run', source / 'queries.tsv'
    for file in (run_path, queries_path):
        if digest(file) != metadata['outputs_sha256'][file.name]:
            raise ValueError(f'Saved baseline checksum does not match: {file}')
    saved_queries = dict(line.split('\t', 1) for line in queries_path.read_text(encoding='utf-8').splitlines())
    if any(saved_queries.get(query.qid) != query.text for query in queries):
        raise ValueError(f'The selected questions differ from the saved {method} run.')

    rankings = {query.qid: [] for query in queries}
    with run_path.open(encoding='utf-8') as stream:
        for line in stream:
            qid, _, doc_id, rank, score, run_method = line.split()
            if qid in rankings:
                hits = rankings[qid]
                rank, score = int(rank), float(score)
                if run_method != method or rank != len(hits) + 1 or not isfinite(score):
                    raise ValueError(f'Invalid saved {method} ranking for question {qid}.')
                hits.append(Hit(doc_id, rank, score, rank, score))
    expected = min(depth, document_count)
    for qid, hits in rankings.items():
        if len(hits) != expected or len({hit.doc_id for hit in hits}) != expected:
            raise ValueError(f'Expected {expected} distinct saved {method} candidates for question {qid}.')
    return rankings, metadata


def remote_batches(query, candidates, texts, batch_size):
    prefix = json.dumps({'query': query}, ensure_ascii=False).encode('utf-8')[:-1] + b',"documents":['
    suffix = b']}'
    limit = 32 * 1024 * 1024 - len(prefix) - len(suffix)
    parts, originals, size = [], {}, 0
    # End each request on the original model batch boundary so padding stays identical.
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        data = b','.join(json.dumps([hit.doc_id, texts[hit.doc_id]], ensure_ascii=False).encode('utf-8')
                         for hit in batch)
        if len(data) > limit:
            raise ValueError('One model batch exceeds the AutoDL request size limit.')
        if parts and size + 1 + len(data) > limit:
            yield prefix + b','.join(parts) + suffix, originals
            parts, originals, size = [], {}, 0
        size += len(data) + bool(parts)
        parts.append(data)
        originals.update((hit.doc_id, hit) for hit in batch)
    if parts:
        yield prefix + b','.join(parts) + suffix, originals


def request_remote_batch(url, address, body, originals, config):
    connection = HTTPConnection(address.hostname, address.port, timeout=180)
    started = perf_counter()
    try:
        connection.request('POST', '/rerank', body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        result = response.read(1024 * 1024 + 1)
    except (OSError, HTTPException) as exc:
        raise RuntimeError(f'Cannot reach the AutoDL reranker through {url}: {exc}') from exc
    finally:
        connection.close()
    request_seconds = perf_counter() - started
    if response.status != 200:
        raise RuntimeError(f'AutoDL reranker returned HTTP {response.status}: {result[:500].decode("utf-8", "replace")}')
    if len(result) > 1024 * 1024:
        raise ValueError('AutoDL reranker response is too large.')
    reply = json.loads(result)
    if not isinstance(reply, dict) or reply.get('config') != config:
        raise ValueError('AutoDL reranker model or settings differ from the local experiment.')
    rows, seconds, software = reply.get('ranking'), reply.get('model_seconds'), reply.get('software')
    if (not isinstance(rows, list) or len(rows) != len(originals)
            or type(seconds) not in {int, float} or not isfinite(seconds) or seconds < 0
            or not isinstance(software, dict)):
        raise ValueError('Invalid AutoDL reranker response.')
    hits, seen = [], set()
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError('Invalid row in AutoDL reranker response.')
        doc_id, score = row
        if (not isinstance(doc_id, str) or doc_id not in originals or doc_id in seen
                or type(score) not in {int, float} or not isfinite(score)):
            raise ValueError('Invalid document ID or score in AutoDL reranker response.')
        seen.add(doc_id)
        hits.append(replace(originals[doc_id], rank=rank, score=float(score), rerank_score=float(score)))
    if hits != sorted(hits, key=lambda hit: (-hit.score, hit.doc_id)):
        raise ValueError('AutoDL returned an invalid score order.')
    proof = sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return hits, request_seconds, seconds, software, proof


def request_remote_ranking(url, query, candidates, texts, config):
    address = urlsplit(url)
    if (address.scheme != 'http' or address.hostname not in {'127.0.0.1', 'localhost'}
            or not address.port or address.path not in {'', '/'} or address.query
            or address.fragment or address.username is not None):
        raise ValueError('Use a local SSH tunnel address such as http://127.0.0.1:16006.')
    hits, proofs = [], []
    request_seconds = model_seconds = 0.0
    software = None
    for body, originals in remote_batches(query, candidates, texts, config['batch_size']):
        batch, request_time, model_time, batch_software, proof = request_remote_batch(
            url, address, body, originals, config)
        if software is not None and software != batch_software:
            raise ValueError('AutoDL reranker software changed during this query.')
        software = batch_software
        hits.extend(batch)
        proofs.append(proof)
        request_seconds += request_time
        model_seconds += model_time
    hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
    ranked = [replace(hit, rank=rank) for rank, hit in enumerate(hits, 1)]
    return ranked, request_seconds, model_seconds, software, proofs


def rerank_candidates(corpus_dir: Path, queries: list[Query], candidates: dict[str, list[Hit]],
                      device: str, depth: int, *, reranker_url: str | None = None):
    cfg = load_config(ROOT / 'configs' / 'default.yaml')
    config = dict(cfg['reranker'], device='cuda' if reranker_url is not None else device,
                  depth=depth, max_length=512)
    if reranker_url is None:
        repo = config['model'].replace('/', '--')
        snapshot = path(cfg['storage']['model_cache']) / 'hub' / f'models--{repo}' / 'snapshots' / config['revision']
        if not snapshot.is_dir():
            raise FileNotFoundError(f'Install the pinned reranker at {snapshot}.')

    started = perf_counter()
    doc_ids = {hit.doc_id for hits in candidates.values() for hit in hits}
    docs = load_candidate_documents(corpus_dir, doc_ids)
    lookup_seconds = perf_counter() - started
    print(f'Loaded {len(docs):,} candidate texts in {lookup_seconds:.2f}s.', flush=True)

    timing = {'document_lookup_seconds': lookup_seconds}
    if reranker_url is None:
        from ..reranker import Reranker

        started = perf_counter()
        reranker = Reranker(docs, dict(config, model=str(snapshot)))
        timing['model_load_seconds'] = perf_counter() - started
    else:
        texts = {doc.doc_id: doc.search_text for doc in docs}
        request_seconds = model_seconds = 0.0
        server_software = None
        remote_proofs = {}

    rankings, times = {}, {}
    for number, query in enumerate(queries, 1):
        before = candidates[query.qid]
        started = perf_counter()
        if reranker_url is None:
            hits = reranker.rerank(query.text, before)
        else:
            hits, request_time, model_time, software, proofs = request_remote_ranking(
                reranker_url, query.text, before, texts, config)
            remote_proofs[query.qid] = proofs
            if server_software is not None and server_software != software:
                raise ValueError('AutoDL reranker software changed during this run.')
            server_software = software
            request_seconds += request_time
            model_seconds += model_time
        times[query.qid] = perf_counter() - started
        if (len(hits) != len(before)
                or {hit.doc_id: (hit.retrieval_rank, hit.retrieval_score) for hit in hits}
                != {hit.doc_id: (hit.rank, hit.score) for hit in before}
                or any(hit.rerank_score != hit.score for hit in hits)):
            raise ValueError(f'Reranking changed the candidate set, original scores or scoring coverage for {query.qid}.')
        rankings[query.qid] = hits
        if number == 1 or number % 25 == 0 or number == len(queries):
            print(f'Reranked {number}/{len(queries)} queries ({sum(times.values()):.2f}s).', flush=True)
    timing['rerank_seconds'] = sum(times.values())
    if reranker_url is not None:
        timing.update(remote_request_seconds=request_seconds, remote_model_seconds=model_seconds,
                      remote_batch_proofs=remote_proofs)
        config.update(backend='autodl', server_software=server_software)
    return rankings, config, times, timing


def compare_rankings(candidates, metrics, judgments, times, method):
    reranked = f'{method}_rerank'
    comparison = {'per_query': {qid: {'rerank_seconds': seconds} for qid, seconds in times.items()}}
    for name, labels in judgments.items():
        before = score_rankings(candidates, labels)
        after = metrics[name]
        comparison[name] = {method: before['overall'], reranked: after['overall']}
        for qid, row in comparison['per_query'].items():
            old, new = before['per_query'][qid], after['per_query'][qid]
            if old['recall@100'] != new['recall@100']:
                raise ValueError(f'Recall@100 changed after reordering the same candidates for {name} {qid}.')
            row[name] = {method: old, reranked: new}
    return comparison


def write_rank_changes(file: Path, rankings: dict[str, list[Hit]], method: str) -> None:
    with file.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(f'qid\tdocid\t{method}_rank\trerank_rank\t{method}_score\trerank_score\n')
        for qid, hits in rankings.items():
            for hit in hits:
                stream.write(f'{qid}\t{hit.doc_id}\t{hit.retrieval_rank}\t{hit.rank}\t'
                             f'{hit.retrieval_score!r}\t{hit.rerank_score!r}\n')
