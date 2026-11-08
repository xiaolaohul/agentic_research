"""Export TREC rankings and score them against the benchmark's official qrels."""
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

from ..dataset import load_qrels
from ..evaluation import evaluate_query
from ..schemas import Hit

QRELS_FILES = {'evidence': 'qrel_evidence.txt', 'gold': 'qrel_golds.txt'}
METRICS = {'ndcg@10': 'ndcg_cut_10', 'recall@100': 'recall_100'}


def load_judgments(qrels_dir: Path, query_ids: set[str]) -> dict[str, dict]:
    judgments = {}
    for name, filename in QRELS_FILES.items():
        labels = load_qrels(qrels_dir / filename)
        if labels.keys() != query_ids:
            raise ValueError(f'Query IDs in {filename} do not match the benchmark questions.')
        if any(set(docs.values()) != {1} for docs in labels.values()):
            raise ValueError(f'Expected binary positive judgments in {filename}.')
        judgments[name] = labels
    return judgments


def write_run(path: Path, rankings: dict[str, list[Hit]], method: str, depth: int) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as file:
        for qid, hits in rankings.items():
            if len(hits) > depth or len({hit.doc_id for hit in hits}) != len(hits):
                raise ValueError(f'Invalid ranking length or duplicate document for query {qid}.')
            for rank, hit in enumerate(hits, 1):
                if hit.rank != rank or not hit.doc_id or any(c.isspace() for c in hit.doc_id) or not isfinite(hit.score):
                    raise ValueError(f'Invalid hit for query {qid} at rank {rank}.')
                file.write(f'{qid} Q0 {hit.doc_id} {hit.rank} {hit.score!r} {method}\n')


def score_rankings(rankings: dict[str, list[Hit]], labels: dict[str, dict[str, int]]) -> dict:
    from pytrec_eval import RelevanceEvaluator

    # trec_eval orders by score. Reciprocal ranks preserve the searcher's order even on ties.
    run = {qid: {hit.doc_id: 1.0 / hit.rank for hit in hits} for qid, hits in rankings.items()}
    scores = RelevanceEvaluator(labels, {'ndcg_cut.10', 'recall.100'}).evaluate(run)
    rows = {}
    max_error = 0.0
    for qid, hits in rankings.items():
        result = scores[qid] if hits else scores.get(qid, {metric: 0.0 for metric in METRICS.values()})
        rows[qid] = {metric: float(result[key]) for metric, key in METRICS.items()}
        # Recalculate against all judgments with the project's independent implementation.
        expected = evaluate_query(hits, labels[qid], set(labels[qid]), (10, 100))
        for metric, value in rows[qid].items():
            reference = expected[metric]
            if not isclose(value, reference, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'Metric mismatch for query {qid}: {metric} {value} != {reference}.')
            max_error = max(max_error, abs(value - reference))
    return {
        'overall': {metric: mean(row[metric] for row in rows.values()) for metric in METRICS},
        'queries_evaluated': len(rows),
        'per_query': rows,
        'validation': {'checks': len(rows) * len(METRICS), 'max_absolute_error': max_error},
    }
