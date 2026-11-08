"""Independent numerical checks using sklearn, without rerunning retrieval or generation."""
import importlib.metadata
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from .dataset import load_documents, load_qrels
from .utils import digest, jsonl, object_hash, read_json, utc_now, write_json


def paired_analysis(run, all_pairs=False):
    run = Path(run)
    cfg, metrics = read_json(run / 'config.json'), read_json(run / 'metrics.json')['per_query']
    k = 10 if 10 in cfg['evaluation']['ks'] else cfg['evaluation']['ks'][0]
    metric = f'ndcg@{k}'
    queries = {r['qid']: r['query'] for r in jsonl(run / 'retrieval.jsonl')}
    comparisons = {}
    from .schemas import HYBRID_METHODS
    methods = tuple(method for method in HYBRID_METHODS if method in metrics)
    pairs = list(combinations(methods, 2)) if all_pairs else [('bm25', 'bge'), ('bm25', 'bm25_rerank'), ('bge', 'bge_rerank')]
    if not all_pairs and 'rrf' in methods:
        pairs += [(base, 'rrf') for base in ('bm25', 'bge', 'bm25_rerank', 'bge_rerank')]
    for base, other in pairs:
        cases = []
        for qid, query in queries.items():
            first, second = metrics[base][qid][metric], metrics[other][qid][metric]
            if first is None or second is None:
                continue
            cases.append({'qid': qid, 'query': query, 'base_value': first, 'other_value': second,
                          'delta': second - first, 'report': 'report/q-' + object_hash(qid)[:16] + '.html'})
        comparisons[base + '__' + other] = {
            'base': base, 'other': other, 'valid_queries': len(cases),
            'excluded_queries': len(queries) - len(cases),
            'base_mean': mean(r['base_value'] for r in cases) if cases else None,
            'other_mean': mean(r['other_value'] for r in cases) if cases else None,
            'mean_delta': mean(r['delta'] for r in cases) if cases else None,
            'median_delta': median(r['delta'] for r in cases) if cases else None,
            'wins': sum(r['delta'] > 1e-12 for r in cases), 'losses': sum(r['delta'] < -1e-12 for r in cases),
            'ties': sum(abs(r['delta']) <= 1e-12 for r in cases),
            'cases': sorted(cases, key=lambda r: r['delta'], reverse=True)}
    result = {'metric': metric, 'ndcg_ideal_scope': cfg['evaluation'].get('ndcg_scope', 'all_judgments'),
              'comparisons': comparisons,
              'interpretation': 'Descriptive comparisons on these queries only, not a statistical significance test.'}
    write_json(run / 'paired_analysis.json', result)
    return result


def validate_metrics(run):
    import numpy as np
    from sklearn.metrics import ndcg_score, recall_score, precision_score
    run = Path(run)
    cfg, metrics = read_json(run / 'config.json'), read_json(run / 'metrics.json')
    judgments = load_qrels(run / 'qrels.tsv')
    corpus_ids = {doc.doc_id for doc in load_documents(run / 'corpus.jsonl')}
    ndcg_scope = cfg['evaluation'].get('ndcg_scope', 'all_judgments')
    if ndcg_scope not in ('corpus', 'all_judgments'):
        raise ValueError('ndcg_scope must be corpus or all_judgments')
    counts, errors = {'ndcg': 0, 'recall': 0, 'precision': 0}, []
    maximum = 0.0
    for row in jsonl(run / 'retrieval.jsonl'):
        labels = judgments.get(row['qid'], {})
        for method, result in row['methods'].items():
            hits = result['hits']
            ids = sorted(set(labels) | {h['doc_id'] for h in hits})
            if len(ids) < 2:
                continue
            rank = {h['doc_id']: h['rank'] for h in hits}
            gains = np.array([max(0, labels.get(doc, 0))
                              if ndcg_scope == 'all_judgments' or doc in corpus_ids else 0
                              for doc in ids], dtype=float)
            if cfg['evaluation']['gain'] != 'linear':
                gains = np.power(2.0, gains) - 1.0
            scores = np.array([1.0 / rank[doc] if doc in rank else 0.0 for doc in ids])
            # Missing result slots have zero gain; do not let unretrieved judged items
            # drift into the DCG cutoff through tied prediction scores of zero.
            padding = range(len(hits) + 1, max(cfg['evaluation']['ks']) + 1)
            ndcg_gains = np.concatenate([gains, np.zeros(len(padding))])
            ndcg_scores = np.concatenate([scores, np.array([1.0 / i for i in padding])])
            relevant = np.array([labels.get(doc, 0) >= cfg['evaluation']['relevant_threshold'] for doc in ids])
            if relevant.any():
                predicted5 = np.array([doc in rank and rank[doc] <= 5 for doc in ids])
                # sklearn divides by returned positives; extend to a fixed five slots.
                value = float(precision_score(relevant, predicted5, zero_division=0)) * int(predicted5.sum()) / 5
                reported = metrics['per_query'][method][row['qid']]['precision@5']
                error = abs(value - reported) if reported is not None else float('inf')
                counts['precision'] += 1
                maximum = max(maximum, error)
                if error > 1e-10:
                    errors.append({'qid': row['qid'], 'method': method, 'metric': 'precision@5',
                                   'expected': value, 'reported': reported})
            for k in cfg['evaluation']['ks']:
                expected = {}
                if gains.sum() > 0:
                    expected[f'ndcg@{k}'] = float(ndcg_score(ndcg_gains[None, :], ndcg_scores[None, :], k=k, ignore_ties=True))
                if relevant.any():
                    predicted = np.array([doc in rank and rank[doc] <= k for doc in ids])
                    expected[f'recall@{k}'] = float(recall_score(relevant, predicted))
                for key, value in expected.items():
                    reported = metrics['per_query'][method][row['qid']][key]
                    error = abs(value - reported) if reported is not None else float('inf')
                    counts[key.split('@')[0]] += 1
                    maximum = max(maximum, error)
                    if error > 1e-10:
                        errors.append({'qid': row['qid'], 'method': method, 'metric': key,
                                       'expected': value, 'reported': reported})
    report = {'checked_at': utc_now(), 'provider': 'scikit-learn',
              'ndcg_ideal_scope': ndcg_scope,
              'corpus_sha256': digest(run / 'corpus.jsonl'),
              'provider_version': importlib.metadata.version('scikit-learn'), 'checks': counts,
              'passed': not errors, 'max_absolute_error': maximum if not errors else None,
              'qrels_sha256': digest(run / 'qrels.tsv'), 'retrieval_sha256': digest(run / 'retrieval.jsonl'),
              'errors': errors}
    write_json(run / 'metrics_validation.json', report)
    if errors:
        raise ValueError(f'独立指标核验失败：{len(errors)} 项，请查看 metrics_validation.json。')
    print(f'Independent metrics validation passed: {counts}', flush=True)
    return report
