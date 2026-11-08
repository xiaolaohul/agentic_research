import tempfile
from pathlib import Path
import unittest
from src.cohorts import cohort_panel
from src.schemas import METHODS
from src.utils import ROOT, write_json, read_json


class CohortTests(unittest.TestCase):
    def test_separate_cohort_metrics_use_each_topic_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache/test-tmp') as folder:
            run=Path(folder)
            write_json(run/'cohorts.json',{'note':'two independent batches','cohorts':[
                {'name':'old','query_ids':['a']},{'name':'new','query_ids':['b']}]})
            records=[{'qid':qid,'methods':{m:{'hits':[{'doc_id':'doc','rank':1}],
                      'retrieval_ms':1,'rerank_ms':0,'total_ms':1} for m in METHODS}} for qid in ['a','b']]
            panel=cohort_panel(run,records,{'a':{'doc':2},'b':{'other':1}},
                               {'doc'},{'ks':[5,10,100],'relevant_threshold':1,'gain':'linear'})
            stats=read_json(run/'cohort_metrics.json')
            self.assertEqual(stats['old']['aggregate']['bm25']['precision@5']['mean'],.2)
            self.assertEqual(stats['new']['aggregate']['bm25']['precision@5']['mean'],0)
            self.assertEqual(stats['全部100题']['aggregate']['bm25']['precision@5']['mean'],.1)
            self.assertIn('two independent batches',panel)
