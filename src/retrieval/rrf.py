"""Equal-weight reciprocal rank fusion; source scores never affect ordering."""
import time


def fuse(rankings, rank_constant=60, input_k=100, output_k=100):
    for value in (rank_constant, input_k, output_k):
        if type(value) is not int or value < 1:
            raise ValueError('RRF常数、输入深度和输出数量必须是正整数。')
    contributions = {}
    for method in ('bm25', 'bge'):
        hits = rankings[method]['hits']
        if [hit['rank'] for hit in hits] != list(range(1, len(hits) + 1)):
            raise ValueError('RRF输入必须按连续的原始名次排列。')
        ids = [hit['doc_id'] for hit in hits]
        if len(ids) != len(set(ids)):
            raise ValueError('RRF输入存在重复段落ID。')
        for hit in hits[:input_k]:
            contributions.setdefault(hit['doc_id'], {})[method] = {
                'rank': hit['rank'], 'score': hit['score'],
                'contribution': 1.0 / (rank_constant + hit['rank'])}
    scores = {doc: sum(item['contribution'] for item in sources.values())
              for doc, sources in contributions.items()}
    ordered = sorted(scores, key=lambda doc: (-scores[doc], doc))[:output_k]
    hits = []
    for rank, doc in enumerate(ordered, 1):
        source = contributions[doc]
        hits.append({'doc_id': doc, 'rank': rank, 'score': scores[doc],
                     'retrieval_rank': rank, 'retrieval_score': scores[doc], 'rerank_score': None,
                     'bm25_rank': source.get('bm25', {}).get('rank'),
                     'bge_rank': source.get('bge', {}).get('rank'),
                     'bm25_contribution': source.get('bm25', {}).get('contribution', 0.0),
                     'bge_contribution': source.get('bge', {}).get('contribution', 0.0),
                     'fusion_sources': source})
    return hits, len(contributions)


def fused_result(methods, config, reused=False):
    start = time.perf_counter()
    hits, union_count = fuse(methods, config['rank_constant'], config['input_k'], config['output_k'])
    fusion_ms = (time.perf_counter() - start) * 1000
    retrieval_ms = sum(methods[method]['retrieval_ms'] for method in ('bm25', 'bge'))
    return {'hits': hits, 'retrieval_ms': retrieval_ms, 'fusion_ms': fusion_ms,
            'rerank_ms': 0.0, 'total_ms': retrieval_ms + fusion_ms,
            'candidate_union_count': union_count, 'rerank_depth': 0, 'reranked_count': 0,
            'timing_kind': 'historical_serial_estimate' if reused else 'serial_stage_sum',
            'timing_note': 'BM25耗时+BGE耗时+本次融合耗时；模型加载不计入。'}
