"""Graded nDCG with explicit denominators, missing judgments and subset coverage."""
import math
from statistics import mean


def ndcg_description(cfg):
    if cfg.get('ndcg_scope', 'all_judgments') == 'corpus':
        return ('nDCG 理想排序仅使用当前固定语料内、针对当前问题的正等级标注；'
                '各方案共用同一理想排序，不以各自检索出的候选集作基准。'
                '该口径用于子集实验，不直接与完整语料评测分数比较。')
    return 'nDCG 理想排序使用当前问题全部原始正等级标注，包含子集之外的段落。'


def evaluate_query(hits, judgments, corpus_ids, ks, threshold=1, gain='linear', ndcg_scope='all_judgments'):
    if ndcg_scope not in ('corpus', 'all_judgments'):
        raise ValueError('ndcg_scope must be corpus or all_judgments')
    ids = [h['doc_id'] if isinstance(h, dict) else h.doc_id for h in hits]
    if len(ids) != len(set(ids)):
        raise ValueError('检索结果包含重复 ID，不能可靠评价。')
    relevant = {doc for doc, grade in judgments.items() if grade >= threshold}
    available = relevant & corpus_ids
    graded = sorted((grade for doc, grade in judgments.items()
                     if grade > 0 and (ndcg_scope == 'all_judgments' or doc in corpus_ids)), reverse=True)
    gain_value = (lambda g: float(g)) if gain == 'linear' else (lambda g: 2.0**g - 1)
    result = {'judged_query': bool(judgments), 'relevant_total': len(relevant),
              'relevant_in_corpus': len(available),
              'corpus_relevant_coverage': len(available) / len(relevant) if relevant else None}
    # Fixed cutoff; absent results and unjudged passages do not inflate precision.
    result['precision@5'] = sum(doc in relevant for doc in ids[:5]) / 5 if relevant else None
    for k in ks:
        top = ids[:k]
        found = sum(doc in relevant for doc in top)
        first = next((rank for rank, doc in enumerate(top, 1) if doc in relevant), None)
        dcg = sum(gain_value(max(0, judgments.get(doc, 0))) / math.log2(rank + 1)
                  for rank, doc in enumerate(top, 1))
        ideal = sum(gain_value(g) / math.log2(rank + 1) for rank, g in enumerate(graded[:k], 1))
        result[f'ndcg@{k}'] = dcg / ideal if ideal else None
        result[f'mrr@{k}'] = (1.0 / first if first else 0.0) if relevant else None
        result[f'recall@{k}'] = found / len(relevant) if relevant else None
        # Divide by K even for short runs; no inflation when the corpus has fewer than K docs.
        result[f'judged@{k}'] = sum(doc in judgments for doc in top) / k
        result[f'returned@{k}'] = len(top)
    return result


def evaluate(records, labels, corpus_ids, cfg):
    from .schemas import methods_for
    methods = methods_for(records)
    per_query, aggregate = {}, {}
    for method in methods:
        rows = {}
        for row in records:
            rows[row['qid']] = evaluate_query(row['methods'][method]['hits'], labels.get(row['qid'], {}),
                corpus_ids, cfg['ks'], cfg['relevant_threshold'], cfg['gain'], cfg.get('ndcg_scope', 'all_judgments'))
        per_query[method] = rows
        keys = next(iter(rows.values())).keys() if rows else []
        aggregate[method] = {}
        for key in keys:
            values = [r[key] for r in rows.values() if r[key] is not None]
            if key == 'judged_query':
                continue
            aggregate[method][key] = {'mean': mean(values) if values else None, 'valid_queries': len(values)}
        aggregate[method]['query_count'] = len(rows)
        aggregate[method]['judged_query_count'] = sum(r['judged_query'] for r in rows.values())
        for stage in ('retrieval_ms', 'rerank_ms', 'total_ms'):
            aggregate[method][stage] = {'mean': mean(r['methods'][method][stage] for r in records),
                                       'valid_queries': len(records)}
        if any('fusion_ms' in r['methods'][method] for r in records):
            aggregate[method]['fusion_ms'] = {'mean': mean(r['methods'][method].get('fusion_ms', 0.0) for r in records),
                                               'valid_queries': len(records)}
    return {'policy': {'qrels_scope': 'all original judgments for selected queries',
                       'unjudged': 'zero gain in metrics; shown separately in report',
                       'queries_without_positives': 'N/A; excluded, counted explicitly',
                       'ndcg_gain': cfg['gain'], 'relevant_threshold': cfg['relevant_threshold'],
                       'ndcg_ideal_scope': cfg.get('ndcg_scope', 'all_judgments'),
                       'ndcg_ideal_description': ndcg_description(cfg),
                       'precision@5': 'relevant passages in top 5 / 5; missing ranks count as non-relevant'},
            'aggregate': aggregate, 'per_query': per_query}
