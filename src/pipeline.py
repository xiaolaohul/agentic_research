from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv
import io
import re
import shutil
import time

from .dataset import validate_prepared
from .evaluation import evaluate
from .schemas import METHODS, HYBRID_METHODS
from .utils import (atomic_text, digest, jsonl, object_hash, path, read_json, release_gpu,
                    storage_check, utc_now, versions, write_json, write_jsonl)


def index_fingerprint(cfg, manifest, method):
    return object_hash({'corpus': manifest['corpus_sha256'], 'method': method,
                        'config': cfg['retrieval'][method], 'format': 1})


def index_directory(cfg, method):
    return path(cfg['retrieval']['index_dir']) / method


def build_indexes(cfg, methods=('bm25', 'bge'), force=False):
    from .retrieval.bm25 import BM25Retriever
    from .retrieval.bge import BGERetriever
    docs, _, _, manifest = validate_prepared(cfg)
    storage_check(cfg, len(docs) * 768 * 4 if 'bge' in methods else 0)
    for method in methods:
        directory = index_directory(cfg, method)
        fingerprint = index_fingerprint(cfg, manifest, method)
        meta = directory / 'metadata.json'
        if not force and meta.exists() and read_json(meta).get('fingerprint') == fingerprint:
            print(f'Index {method}: cache hit.', flush=True)
            continue
        print(f'Index {method}: {len(docs)} segments...', flush=True)
        retriever = (BM25Retriever if method == 'bm25' else BGERetriever)(docs, cfg['retrieval'][method])
        # Metadata is written last. A failed rebuild cannot masquerade as a valid index.
        meta.unlink(missing_ok=True)
        retriever.build(directory, fingerprint)
        del retriever
        release_gpu()


def retrieve(cfg, run_id=None):
    from .retrieval.bm25 import BM25Retriever
    from .retrieval.bge import BGERetriever
    from .reranker import Reranker
    docs, queries, labels, manifest = validate_prepared(cfg)
    hybrid = cfg['retrieval'].get('rrf', {}).get('enabled', False)
    methods = HYBRID_METHODS if hybrid else METHODS
    run_id = run_id or datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('实验编号只能包含字母、数字、下划线和连字符。')
    run = path(cfg['results_dir']) / run_id
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'config.json', cfg)
    write_json(run / 'dataset.json', manifest)
    for key, target in [('corpus', 'corpus.jsonl'), ('queries', 'queries.jsonl'), ('qrels', 'qrels.tsv')]:
        shutil.copyfile(path(cfg['dataset'][key]), run / target)
    state = {'run_id': run_id, 'created_at': utc_now(), 'status': 'running', 'versions': versions(),
             'report_mode': 'retrieval_only', 'reranker_revision': None, 'methods': list(methods),
             'rerank_depth': cfg['reranker'].get('depth', cfg['retrieval']['candidate_k'])}
    write_json(run / 'run.json', state)
    records = {q.qid: {'qid': q.qid, 'query': q.text, 'methods': {}} for q in queries}
    try:
        for method, cls in [('bm25', BM25Retriever), ('bge', BGERetriever)]:
            print(f'Retrieve {method}...', flush=True)
            retriever = cls(docs, cfg['retrieval'][method])
            retriever.load(index_directory(cfg, method), index_fingerprint(cfg, manifest, method))
            if method == 'bge':
                retriever.load_model()  # Model loading is excluded from per-query latency.
            stage_rows = []
            for q in queries:
                start = time.perf_counter()
                hits = retriever.retrieve(q.text, cfg['retrieval']['candidate_k'])
                elapsed = (time.perf_counter() - start) * 1000
                records[q.qid]['methods'][method] = {'hits': [h.to_dict() for h in hits],
                    'retrieval_ms': elapsed, 'rerank_ms': 0.0, 'total_ms': elapsed}
                stage_rows.append({'qid': q.qid, **records[q.qid]['methods'][method]})
            write_jsonl(run / f'{method}.candidates.jsonl', stage_rows)
            if method == 'bge':
                state['bge_revision'] = read_json(index_directory(cfg, method) / 'metadata.json').get('resolved_revision')
            del retriever
            release_gpu()
        if hybrid:
            from .retrieval.rrf import fused_result
            for row in records.values():
                row['methods']['rrf'] = fused_result(row['methods'], cfg['retrieval']['rrf'])
            write_jsonl(run / 'rrf.candidates.jsonl',
                        ({'qid': row['qid'], **row['methods']['rrf']} for row in records.values()))
        print('Rerank both candidate sets with the SAME model...', flush=True)
        reranker = Reranker(docs, cfg['reranker'])
        state['reranker_revision'] = reranker.resolved_revision
        from .schemas import Hit
        for q in queries:
            for base in ('bm25', 'bge'):
                original = records[q.qid]['methods'][base]
                start = time.perf_counter()
                hits = reranker.rerank(q.text, [Hit(**h) for h in original['hits']])
                elapsed = (time.perf_counter() - start) * 1000
                records[q.qid]['methods'][base + '_rerank'] = {'hits': [h.to_dict() for h in hits],
                    'retrieval_ms': original['retrieval_ms'], 'rerank_ms': elapsed,
                    'total_ms': original['retrieval_ms'] + elapsed,
                    'rerank_depth': state['rerank_depth'],
                    'reranked_count': min(state['rerank_depth'], len(original['hits']))}
        del reranker
        release_gpu()
        rows = list(records.values())
        write_jsonl(run / 'retrieval.jsonl', rows)
        metrics = evaluate(rows, labels, {d.doc_id for d in docs}, cfg['evaluation'])
        write_json(run / 'metrics.json', metrics)
        export_metrics_csv(run, metrics)
        for method in methods:
            # Rank-only scores guarantee the same ordering when external evaluators break raw-score ties.
            atomic_text(run / f'{method}.trec', ''.join(
                f"{r['qid']} Q0 {h['doc_id']} {h['rank']} {1.0 / h['rank']:.12f} {method}\n"
                for r in rows for h in r['methods'][method]['hits']))
        state.update(status='complete', completed_at=utc_now())
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(run / 'run.json', state)
    print(f'Results: {run}', flush=True)
    return run


def export_metrics_csv(run, metrics):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['method', 'metric', 'mean', 'valid_queries'])
    for method, stats in metrics['aggregate'].items():
        for key, value in stats.items():
            if isinstance(value, dict):
                writer.writerow([method, key, value['mean'], value['valid_queries']])
    atomic_text(run / 'metrics.csv', '\ufeff' + stream.getvalue())


def resolve_run(cfg, value=None):
    if value:
        candidate = path(value)
        if not candidate.exists():
            candidate = path(cfg['results_dir']) / value
    else:
        runs = sorted(path(cfg['results_dir']).glob('*/run.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        runs = [p for p in runs if read_json(p).get('status') == 'complete']
        if not runs:
            raise ValueError('没有完成的实验，请先运行 run。')
        candidate = runs[0].parent
    if read_json(candidate / 'run.json')['status'] != 'complete':
        raise ValueError('实验尚未完整完成，不能生成完整比较报告。')
    return candidate
