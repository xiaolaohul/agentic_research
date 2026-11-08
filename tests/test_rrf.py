import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.dataset import save_prepared
from src.evaluation import evaluate
from src.hybrid import supplement_rrf
from src.reranker import Reranker
from src.retrieval.rrf import fuse
from src.schemas import Document, Query, Hit, METHODS, HYBRID_METHODS, methods_for
from src.utils import read_json, write_json, write_jsonl, jsonl, digest
from tests.test_core import temporary_directory


def hits(ids):
    return [Hit(doc, rank, 1000.0/rank, rank, 1000.0/rank).to_dict()
            for rank, doc in enumerate(ids, 1)]


class RRFTests(unittest.TestCase):
    def test_overlap_missing_branch_and_source_scores(self):
        rankings = {'bm25': {'hits':hits(['a','b'])}, 'bge': {'hits':hits(['c','b'])}}
        fused, union = fuse(rankings, 60, 2, 3)
        self.assertEqual(union, 3)
        self.assertEqual([h['doc_id'] for h in fused], ['b','a','c'])
        self.assertAlmostEqual(fused[0]['score'], 2/62)
        self.assertEqual(fused[1]['bge_rank'], None)
        self.assertEqual(fused[1]['bge_contribution'], 0)
        self.assertEqual(fused[0]['bm25_rank'], 2)
        altered = copy.deepcopy(rankings)
        altered['bm25']['hits'][0]['score'] = -1e20
        self.assertEqual([h['doc_id'] for h in fuse(altered,60,2,3)[0]], ['b','a','c'])
        self.assertEqual(len(fuse(rankings,60,2,2)[0]), 2)

    def test_input_cutoff_duplicates_parameters_and_method_consistency(self):
        rankings = {'bm25': {'hits':hits(['b','a'])}, 'bge': {'hits':hits(['c','a'])}}
        self.assertEqual([h['doc_id'] for h in fuse(rankings,60,1,10)[0]], ['b','c'])
        rankings['bm25']['hits'] = hits(['a','a'])
        with self.assertRaisesRegex(ValueError, '重复'):
            fuse(rankings)
        with self.assertRaises(ValueError):
            fuse(rankings, rank_constant=0)
        with self.assertRaises(ValueError):
            methods_for([{'methods':{'bm25':{}}}, {'methods':{'bge':{}}}])

    def test_depth_only_scores_selected_prefix_and_preserves_tail(self):
        ranker = Reranker.__new__(Reranker)
        ranker.config = {'depth':2, 'batch_size':4, 'device':'cpu', 'max_length':32}
        ranker.docs = {doc:Document(doc, doc) for doc in ['a','b','c']}
        calls = []
        class Tokens(dict):
            def to(self, device): return self
        def tokenize(pairs, **kwargs):
            calls.extend(pairs)
            return Tokens()
        class Scores:
            def reshape(self, *args): return self
            def float(self): return self
            def cpu(self): return self
            def tolist(self): return [1,5]
        ranker.tokenizer = tokenize
        ranker.torch = SimpleNamespace(inference_mode=nullcontext)
        ranker.model = lambda **kwargs: SimpleNamespace(logits=Scores())
        result = ranker.rerank('question', [Hit(**h) for h in hits(['a','b','c'])])
        self.assertEqual([h.doc_id for h in result], ['b','a','c'])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(result[-1].rerank_score)

    def test_supplement_and_legacy_report_round_trip(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = root/'original'
            source.mkdir()
            cfg = {'dataset': {'name':'fixture','corpus':str(source/'corpus.jsonl'),
                    'queries':str(source/'queries.jsonl'),'qrels':str(source/'qrels.tsv'),
                    'manifest':str(source/'dataset.json'),'seed':42},
                'storage':{'reserve_gib':0,'budget_gib':100000}, 'results_dir':str(root/'results'),
                'retrieval':{'candidate_k':2,'context_k':1,'display_k':2,'index_dir':'indexes',
                             'bm25':{},'bge':{'model':'fixture'}},
                'reranker':{'model':'fixture','max_length':32},
                'evaluation':{'ks':[1,2],'gain':'linear','relevant_threshold':1,'ndcg_scope':'corpus'}}
            labels = {'sample-1':{'sample-a':3,'sample-b':0,'sample-c':2}}
            save_prepared(cfg, [Document('sample-'+d,d) for d in ['a','b','c']],
                          [Query('sample-1','question')], labels, {'kind':'synthetic'})
            write_json(source/'config.json',cfg)
            write_json(source/'run.json',{'status':'complete','run_id':'original','versions':{},
                                         'generation_status':'not_requested','report_mode':'retrieval_only'})
            data = {}
            for method in METHODS:
                data[method] = {'hits':hits(['sample-a','sample-b']),
                                'retrieval_ms':1.,'rerank_ms':2. if method.endswith('rerank') else 0.,
                                'total_ms':3. if method.endswith('rerank') else 1.}
                if method.endswith('rerank'):
                    for h in data[method]['hits']: h['rerank_score'] = h['score']
            records = [{'qid':'sample-1','query':'question','methods':data}]
            write_jsonl(source/'retrieval.jsonl', records)
            old_metrics = evaluate(records,labels,{'sample-a','sample-b','sample-c'},cfg['evaluation'])
            write_json(source/'metrics.json', old_metrics)
            cfg = copy.deepcopy(cfg)
            cfg['reranker']['depth'] = 2
            cfg['retrieval']['rrf'] = {'enabled':True,'rank_constant':60,'input_k':2,'output_k':2}
            original_hash = digest(source/'retrieval.jsonl')
            run = supplement_rrf(source, cfg, 'hybrid')
            self.assertEqual(digest(source/'retrieval.jsonl'),original_hash)
            self.assertEqual(read_json(run/'run.json')['status'],'complete')
            new_records = list(jsonl(run/'retrieval.jsonl'))
            self.assertEqual(methods_for(new_records),HYBRID_METHODS)
            self.assertEqual(new_records[0]['methods']['bm25_rerank']['reranked_count'],2)
            self.assertEqual(read_json(run/'metrics.json')['aggregate']['bm25'],old_metrics['aggregate']['bm25'])
            self.assertEqual(read_json(run/'metrics_validation.json')['checks'],{'ndcg':10,'recall':10,'precision':5})
            text = (run/'report/index.html').read_text(encoding='utf-8')
            self.assertIn('5种检索方案对比', text)
            self.assertIn('RRF', text)
            detail = next((run/'report').glob('q-*.html')).read_text(encoding='utf-8')
            self.assertIn('BM25原名次', detail)
            self.assertIn('--method-count:5', detail)
            self.assertIn('重排深度 2', detail)
            from src.report import generate_report
            generate_report(source)
            self.assertNotIn('RRF', (source/'report/index.html').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                supplement_rrf(source,cfg,'hybrid')


if __name__ == '__main__':
    unittest.main()
