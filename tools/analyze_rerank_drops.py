"""Inspect existing rankings and tokenization; never run retrieval or generation."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rerank_drops import extract_cases, provenance
from src.utils import read_json, write_json, digest, utc_now

RUN = ROOT / 'results/trec24-100q-20261029'
PACKET = ROOT / '.cache/diagnostics/rerank-drop10-evidence.json'


def extract():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer, logging
    logging.set_verbosity_error()
    cfg = read_json(RUN / 'config.json')
    rc = cfg['reranker']
    tok = AutoTokenizer.from_pretrained(rc['model'], revision=rc['revision'],
                                      cache_dir=str(ROOT / cfg['storage']['model_cache'] / 'hub'), local_files_only=True)
    cases = extract_cases(RUN)
    cache = {}
    for c in cases:
        for d in c['changes']:
            key = c['qid'] + '|' + d['doc_id']
            if key not in cache:
                full = tok(c['query'], d['search_text'], truncation=False)
                short = tok(c['query'], d['search_text'], truncation='only_second',
                            max_length=rc['max_length'], return_offsets_mapping=True)
                offsets = [offset for seq, offset in zip(short.sequence_ids(), short['offset_mapping']) if seq == 1]
                cache[key] = {'full_pair_tokens': len(full['input_ids']), 'input_tokens': len(short['input_ids']),
                              'truncated': len(full['input_ids']) > rc['max_length'],
                              'retained_text_end': max((offset[1] for offset in offsets), default=0)}
            d['tokenization'] = cache[key]
    write_json(PACKET, {'provenance': provenance(RUN), 'cases': cases,
                        'tokenizer': {'model': rc['model'], 'revision': rc['revision'], 'max_length': rc['max_length']}})
    print({'cases': len(cases), 'question_document_pairs_checked': len(cache),
           'truncated_pairs': sum(d['truncated'] for d in cache.values())})


def build():
    from collections import Counter
    from tools.rerank_drop_notes import NOTES, NOTES_BGE, QUESTIONS, CATEGORIES
    raw = read_json(PACKET)
    if raw['provenance'] != provenance(RUN):
        raise ValueError('Evidence inputs changed; run --extract again before rebuilding analysis.')
    notes = {**NOTES, **NOTES_BGE}
    assert len(notes) == len(NOTES) + len(NOTES_BGE)
    assert set(notes) == {c['key'] for c in raw['cases']}, 'Every regression needs exactly one reviewed interpretation.'
    cases = raw['cases']
    for c in cases:
        category, secondary, reading, explanation, check = notes[c['key']]
        assert category in CATEGORIES and all(tag in CATEGORIES for tag in secondary)
        assert category not in secondary
        c.update(title=QUESTIONS[c['qid']][0], intent=QUESTIONS[c['qid']][1], category=category,
                 secondary=secondary, reading=reading, explanation=explanation, next_check=check,
                 confidence='原文对照支持的解释；尚未逐项进行因果对照实验')
    diagnostic_path = ROOT / '.cache/diagnostics/2024-36355-rerank.json'
    diagnostic = read_json(diagnostic_path)
    corrected = next(v for v in diagnostic['variants'].values()
                     if v['query'] == diagnostic['variants']['original']['query'].replace('sceant', 'scent'))
    assert diagnostic['reproduction']['same_full_order']
    assert diagnostic['reproduction']['max_score_difference'] == 0
    assert diagnostic['variants']['original']['positive_ranks'] == [6] and corrected['positive_ranks'] == [1]
    c = next(c for c in cases if c['key'] == 'bge|2024-36355')
    c['confidence'] = '已有单词改写对照支持拼写敏感性；真实意图与因果解释仍有条件'
    c['prior_control'] = {'source': str(diagnostic_path.relative_to(ROOT)), 'sha256': digest(diagnostic_path),
                          'original_positive_rank': 6, 'scent_positive_rank': 1,
                          'model_revision': diagnostic['resolved_revision']}
    groups = {}
    for method, expected in [('bm25', 18), ('bge', 32)]:
        selected = [c for c in cases if c['base'] == method]
        assert len(selected) == expected
        counts = Counter(c['category'] for c in selected)
        groups[method] = {'count': len(selected), 'category_counts': dict(counts),
                          'category_questions': {key: [c['qid'] for c in selected if c['category'] == key] for key in counts}}
    unique_inputs = {c['qid'] + '|' + d['doc_id']: d['tokenization'] for c in cases for d in c['changes']}
    output = {'built_at': utc_now(), 'metric': 'ndcg@10', 'ndcg_scope': 'corpus',
              'provenance': provenance(RUN), 'notes_sha256': digest(ROOT / 'tools/rerank_drop_notes.py'),
              'interpretation': 'Ranks, grades, input boundaries and score contributions are verified facts; semantic explanations are hypotheses unless explicitly supported by a control.',
              'tokenizer': raw['tokenizer'], 'input_checks': {'question_document_pairs': len(unique_inputs),
                'truncated_pairs': sum(t['truncated'] for t in unique_inputs.values())},
              'categories': {k: {'title': v[0], 'description': v[1]} for k, v in CATEGORIES.items()},
              'groups': groups, 'cases': cases}
    write_json(RUN / 'rerank_drop10.json', output)
    print({'built': len(cases), 'groups': groups})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true', help='Inspect tokenization only; no model inference')
    parser.add_argument('--build', action='store_true', help='Combine evidence with reviewed per-case interpretations')
    args = parser.parse_args()
    if args.extract or not args.build:
        extract()
    if args.build:
        build()
