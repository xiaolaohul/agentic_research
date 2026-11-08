from pathlib import Path
import unittest

from src.pair_report import pair_comparison_panel
from src.schemas import METHODS
from src.utils import read_json, write_json, write_jsonl
from src.validation import paired_analysis
from tests.test_core import temporary_directory


class PairComparisonTests(unittest.TestCase):
    def fixture(self, run, empty=False):
        values = [[.2, .6, .3, .8], [.4, .1, .4, .2], [.8, .8, .9, .8], [None, .4, .5, .5]]
        if empty:
            values = [[None] * 4 for row in values]
        write_json(run / 'config.json', {'evaluation': {'ks': [5, 10], 'ndcg_scope': 'corpus'}})
        write_json(run / 'metrics.json', {'per_query': {
            method: {str(i): {'ndcg@10': row[j]} for i, row in enumerate(values)}
            for j, method in enumerate(METHODS)}})
        write_jsonl(run / 'retrieval.jsonl', [{'qid': str(i), 'query': '<img src=x onerror=alert(1)>'}
                                              for i in range(len(values))])

    def test_all_six_pairs_use_matching_valid_topics_and_raw_differences(self):
        with temporary_directory() as folder:
            run = Path(folder)
            self.fixture(run)
            result = paired_analysis(run, all_pairs=True)
            self.assertEqual(len(result['comparisons']), 6)
            self.assertEqual(result['ndcg_ideal_scope'], 'corpus')
            pair = result['comparisons']['bm25__bge']
            self.assertEqual((pair['wins'], pair['ties'], pair['losses']), (1, 1, 1))
            self.assertEqual((pair['valid_queries'], pair['excluded_queries']), (3, 1))
            self.assertAlmostEqual(pair['base_mean'], 1.4 / 3)
            self.assertAlmostEqual(pair['other_mean'], .5)
            self.assertAlmostEqual(pair['mean_delta'], .1 / 3)
            self.assertEqual(pair['median_delta'], 0)
            self.assertAlmostEqual(result['comparisons']['bm25_rerank__bge_rerank']['median_delta'], -.05)
            self.assertEqual(read_json(run / 'paired_analysis.json'), result)
            self.assertEqual(len(paired_analysis(run)['comparisons']), 3)

    def test_html_escapes_queries_and_keeps_all_case_links(self):
        with temporary_directory() as folder:
            run = Path(folder)
            self.fixture(run)
            result = paired_analysis(run, all_pairs=True)
            rendered = pair_comparison_panel(result)
            self.assertNotIn('<img src=x', rendered)
            self.assertIn('&lt;img src=x', rendered)
            self.assertEqual(rendered.count('data-pair="'), 6)
            self.assertEqual(rendered.count('data-pair-row '), sum(p['valid_queries'] for p in result['comparisons'].values()))
            self.assertIn('data-status="tie"', rendered)
            self.assertIn('data-status="loss"', rendered)
            self.assertIn('data-status="win"', rendered)
            self.assertIn('href="q-', rendered)
            self.assertNotIn('href="report/q-', rendered)

    def test_no_valid_pairs_show_na_and_do_not_divide_by_zero(self):
        with temporary_directory() as folder:
            run = Path(folder)
            self.fixture(run, empty=True)
            result = paired_analysis(run, all_pairs=True)
            for pair in result['comparisons'].values():
                self.assertIsNone(pair['mean_delta'])
                self.assertEqual(pair['excluded_queries'], 4)
            rendered = pair_comparison_panel(result)
            self.assertIn('没有可比较的有效问题', rendered)
            self.assertIn('N/A', rendered)


if __name__ == '__main__':
    unittest.main()
