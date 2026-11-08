from dataclasses import asdict
from pathlib import Path
import unittest

from src.evaluation import evaluate
from src.rerank_drops import extract_cases, provenance, render_drop_panel
from src.schemas import Document
from src.utils import atomic_text, read_json, write_json, write_jsonl
from tests.test_core import temporary_directory


class RerankDropTests(unittest.TestCase):
    def fixture(self, run):
        cfg = {'evaluation': {'ndcg_scope': 'corpus', 'gain': 'linear', 'ks': [10], 'relevant_threshold': 1}}
        write_json(run / 'config.json', cfg)
        docs = [Document(d, '<script>bad()</script>', d) for d in ['a', 'b', 'unknown', 'zero']]
        write_jsonl(run / 'corpus.jsonl', [asdict(d) for d in docs])
        atomic_text(run / 'qrels.tsv', 'q 0 a 3\nq 0 b 1\nq 0 zero 0\nq 0 outside 3\n')
        labels = {'q': {'a': 3, 'b': 1, 'zero': 0, 'outside': 3}}
        orders = {'bm25': ['a', 'b', 'zero', 'unknown'], 'bm25_rerank': ['unknown', 'b', 'zero', 'a'],
                  'bge': ['b', 'a', 'zero', 'unknown'], 'bge_rerank': ['a', 'b', 'zero', 'unknown']}
        records = [{'qid': 'q', 'query': '<img src=x>', 'methods': {
            m: {'hits': [{'doc_id': d, 'rank': i, 'score': 1/i} for i, d in enumerate(order, 1)],
                'retrieval_ms': 1, 'rerank_ms': 0, 'total_ms': 1} for m, order in orders.items()}}]
        write_jsonl(run / 'retrieval.jsonl', records)
        write_json(run / 'metrics.json', evaluate(records, labels, {d.doc_id for d in docs}, cfg['evaluation']))

    def test_extract_contributions_include_zero_and_unjudged_separately(self):
        with temporary_directory() as folder:
            run = Path(folder)
            self.fixture(run)
            cases = extract_cases(run)
            self.assertEqual(len(cases), 1)
            c = cases[0]
            self.assertEqual(c['base'], 'bm25')
            self.assertAlmostEqual(sum(d['ndcg_contribution_delta'] for d in c['changes']), c['delta'])
            self.assertEqual(c['before_grades'], [3, 1, 0, None])
            self.assertEqual(c['after_grades'], [None, 1, 0, 3])
            self.assertNotIn('outside', {d['doc_id'] for d in c['changes']})

    def test_render_escapes_original_text_and_refuses_stale_inputs(self):
        with temporary_directory() as folder:
            run = Path(folder)
            self.fixture(run)
            c = extract_cases(run)[0]
            c.update(title='<img src=x>', intent='intent', category='relation', secondary=[],
                     reading='original reading', explanation='hypothesis', next_check='check', confidence='inference')
            for d in c['changes']:
                d['tokenization'] = {'full_pair_tokens': 12, 'input_tokens': 12, 'truncated': False,
                                     'retained_text_end': len(d['search_text'])}
            data = {'metric': 'ndcg@10', 'ndcg_scope': 'corpus', 'provenance': provenance(run),
                    'categories': {'relation': {'title': '关系', 'description': 'relation'}},
                    'groups': {'bm25': {'count': 1, 'category_counts': {'relation': 1}},
                               'bge': {'count': 0, 'category_counts': {}}},
                    'input_checks': {'question_document_pairs': 4, 'truncated_pairs': 0}, 'cases': [c]}
            write_json(run / 'rerank_drop10.json', data)
            rendered = render_drop_panel(run)
            self.assertNotIn('<script>bad()', rendered)
            self.assertNotIn('<img src=x>', rendered)
            self.assertIn('&lt;img src=x&gt;', rendered)
            self.assertIn('未标注', rendered)
            self.assertIn('drop-bm25-q', rendered)
            self.assertNotIn('nDCG@5', rendered)
            atomic_text(run / 'qrels.tsv', 'q 0 a 2\n')
            with self.assertRaisesRegex(ValueError, '不一致'):
                render_drop_panel(run)


if __name__ == '__main__':
    unittest.main()
