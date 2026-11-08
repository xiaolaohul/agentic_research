"""Replay saved BrowseComp-Plus rankings with the project's existing reranker."""
from math import isfinite
from pathlib import Path
from time import perf_counter

from .dataset import load_candidate_documents
from .evaluation import score_rankings
from ..schemas import Hit, Query
from ..utils import ROOT, digest, load_config, path, read_json


def load_saved_bm25(source: Path, queries: list[Query], benchmark_revision: str, depth: int):
    metadata = read_json(source / 'run.json')
    if (metadata['method'] != 'bm25' or metadata['query_count'] != 830
            or metadata['download_revisions']['benchmark'] != benchmark_revision
            or metadata['max_hits_per_query'] != depth):
        raise ValueError('Expected the saved full BrowseComp-Plus BM25 baseline at the same query revision and depth.')
    run_path, queries_path = source / 'bm25.run', source / 'queries.tsv'
    for file in (run_path, queries_path):
        if digest(file) != metadata['outputs_sha256'][file.name]:
            raise ValueError(f'Saved baseline checksum does not match: {file}')
    saved_queries = dict(line.split('\t', 1) for line in queries_path.read_text(encoding='utf-8').splitlines())
    if any(saved_queries.get(query.qid) != query.text for query in queries):
        raise ValueError('The selected questions differ from the saved BM25 run.')

    rankings = {query.qid: [] for query in queries}
    with run_path.open(encoding='utf-8') as stream:
        for line in stream:
            qid, _, doc_id, rank, score, method = line.split()
            if qid in rankings:
                hits = rankings[qid]
                rank, score = int(rank), float(score)
                if method != 'bm25' or rank != len(hits) + 1 or not isfinite(score):
                    raise ValueError(f'Invalid saved BM25 ranking for question {qid}.')
                hits.append(Hit(doc_id, rank, score, rank, score))
    for qid, hits in rankings.items():
        if len(hits) != depth or len({hit.doc_id for hit in hits}) != depth:
            raise ValueError(f'Expected {depth} distinct saved BM25 candidates for question {qid}.')
    return rankings, metadata


def rerank_candidates(corpus_dir: Path, queries: list[Query], candidates: dict[str, list[Hit]],
                      device: str, depth: int):
    from ..reranker import Reranker

    cfg = load_config(ROOT / 'configs' / 'default.yaml')
    config = dict(cfg['reranker'], device=device, depth=depth, max_length=512)
    repo = config['model'].replace('/', '--')
    snapshot = path(cfg['storage']['model_cache']) / 'hub' / f'models--{repo}' / 'snapshots' / config['revision']
    if not snapshot.is_dir():
        raise FileNotFoundError(f'Install the pinned reranker at {snapshot}.')

    started = perf_counter()
    doc_ids = {hit.doc_id for hits in candidates.values() for hit in hits}
    docs = load_candidate_documents(corpus_dir, doc_ids)
    lookup_seconds = perf_counter() - started
    print(f'Loaded {len(docs):,} candidate texts in {lookup_seconds:.2f}s.', flush=True)

    started = perf_counter()
    reranker = Reranker(docs, dict(config, model=str(snapshot)))
    load_seconds = perf_counter() - started
    rankings, times = {}, {}
    for number, query in enumerate(queries, 1):
        before = candidates[query.qid]
        started = perf_counter()
        hits = reranker.rerank(query.text, before)
        times[query.qid] = perf_counter() - started
        if (len(hits) != len(before)
                or {hit.doc_id: (hit.retrieval_rank, hit.retrieval_score) for hit in hits}
                != {hit.doc_id: (hit.rank, hit.score) for hit in before}
                or any(hit.rerank_score != hit.score for hit in hits)):
            raise ValueError(f'Reranking changed the candidate set, original scores or scoring coverage for {query.qid}.')
        rankings[query.qid] = hits
        if number == 1 or number % 25 == 0 or number == len(queries):
            print(f'Reranked {number}/{len(queries)} queries ({sum(times.values()):.2f}s).', flush=True)
    timing = {'document_lookup_seconds': lookup_seconds, 'model_load_seconds': load_seconds,
              'rerank_seconds': sum(times.values())}
    return rankings, config, times, timing


def compare_rankings(candidates, metrics, judgments, times):
    comparison = {'per_query': {qid: {'rerank_seconds': seconds} for qid, seconds in times.items()}}
    for name, labels in judgments.items():
        before = score_rankings(candidates, labels)
        after = metrics[name]
        comparison[name] = {'bm25': before['overall'], 'bm25_rerank': after['overall']}
        for qid, row in comparison['per_query'].items():
            old, new = before['per_query'][qid], after['per_query'][qid]
            if old['recall@100'] != new['recall@100']:
                raise ValueError(f'Recall@100 changed after reordering the same candidates for {name} {qid}.')
            row[name] = {'bm25': old, 'bm25_rerank': new}
    return comparison


def write_rank_changes(file: Path, rankings: dict[str, list[Hit]]) -> None:
    with file.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('qid\tdocid\tbm25_rank\trerank_rank\tbm25_score\trerank_score\n')
        for qid, hits in rankings.items():
            for hit in hits:
                stream.write(f'{qid}\t{hit.doc_id}\t{hit.retrieval_rank}\t{hit.rank}\t'
                             f'{hit.retrieval_score!r}\t{hit.rerank_score!r}\n')
