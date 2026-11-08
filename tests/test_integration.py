"""Offline integration with explicitly untrained tiny models, never benchmark artifacts."""
from dataclasses import asdict
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from src.dataset import save_prepared
from src.pipeline import build_indexes, retrieve
from src.report import generate_report
from src.schemas import Document, Query, METHODS
from src.utils import ROOT, jsonl, read_json, write_json, digest
from tests.test_core import temporary_directory


class TinyPipelineTests(unittest.TestCase):
    def test_retrieval_pipeline_and_report(self):
        import torch
        from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
        from sentence_transformers import SentenceTransformer, models
        torch.manual_seed(42)
        with temporary_directory() as folder:
            root = Path(folder)
            bert = root / 'bert'
            bert.mkdir()
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'solar', 'panels', 'generate',
                     'electricity', 'sunlight', 'wind', 'turbines', 'plants', 'water', 'store', 'battery',
                     'how', 'do', 'energy', '.', '?', 'a', 'the', 'from', 'and']
            (bert / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
            tokenizer = BertTokenizer(str(bert / 'vocab.txt'), do_lower_case=True)
            tokenizer.save_pretrained(bert)
            model_cfg = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)
            BertModel(model_cfg).save_pretrained(bert)
            embedding = SentenceTransformer(modules=[models.Transformer(str(bert)), models.Pooling(24)])
            embedding_dir = root / 'embedding'
            embedding.save(str(embedding_dir))
            ranking_dir = root / 'reranker'
            model_cfg.num_labels = 1
            BertForSequenceClassification(model_cfg).save_pretrained(ranking_dir)
            tokenizer.save_pretrained(ranking_dir)
            del embedding
            cfg = {
                'dataset': {'name': 'untrained integration fixture', 'corpus': str(root / 'corpus.jsonl'),
                    'queries': str(root / 'queries.jsonl'), 'qrels': str(root / 'qrels.tsv'),
                    'manifest': str(root / 'manifest.json'), 'seed': 42, 'judgments': 'synthetic'},
                'retrieval': {'candidate_k': 3, 'context_k': 2, 'display_k': 2, 'index_dir': str(root / 'indexes'),
                    'bm25': {'k1': 0.9, 'b': 0.4, 'stem': True},
                    'bge': {'model': str(embedding_dir), 'revision': 'main', 'device': 'cpu', 'max_length': 64,
                            'batch_size': 2, 'query_instruction': ''}},
                'reranker': {'model': str(ranking_dir), 'revision': 'main', 'device': 'cpu', 'max_length': 64,
                    'batch_size': 2, 'fp16': False},
                'evaluation': {'ks': [1, 2, 3], 'relevant_threshold': 1, 'gain': 'linear'},
                'storage': {'reserve_gib': 0, 'budget_gib': 100000}, 'results_dir': str(root / 'results'),
            }
            docs = [Document('sample-solar', 'Solar panels generate electricity from sunlight.', '<script>alert(1)</script>'),
                    Document('sample-wind', 'Wind turbines generate electricity.', url='javascript:alert(1)'),
                    Document('sample-battery', 'A battery can store energy.')]
            save_prepared(cfg, docs, [Query('sample-1', 'How do solar panels generate electricity?')],
                          {'sample-1': {'sample-solar': 3, 'outside': 1}}, {'kind': 'synthetic'})
            build_indexes(cfg)
            run = retrieve(cfg, 'integration-fixture')
            records = list(jsonl(run / 'retrieval.jsonl'))
            # A fresh five-method run reuses the same indexes, not just cached rankings.
            hybrid_cfg = copy.deepcopy(cfg)
            hybrid_cfg['retrieval']['rrf'] = {'enabled': True, 'rank_constant': 60, 'input_k': 3, 'output_k': 3}
            hybrid_cfg['reranker']['depth'] = 2
            hybrid_run = retrieve(hybrid_cfg, 'integration-hybrid')
            hybrid_records = list(jsonl(hybrid_run / 'retrieval.jsonl'))
            self.assertEqual(set(hybrid_records[0]['methods']), set(METHODS) | {'rrf'})
            self.assertEqual(hybrid_records[0]['methods']['bm25_rerank']['reranked_count'], 2)
            self.assertEqual(hybrid_records[0]['methods']['bge_rerank']['rerank_depth'], 2)
            self.assertIsNone(hybrid_records[0]['methods']['bge_rerank']['hits'][-1]['rerank_score'])
            self.assertEqual(len(hybrid_records[0]['methods']['rrf']['hits']), 3)
            generate_report(hybrid_run, verify_metrics=True)
            self.assertEqual(set(records[0]['methods']), set(METHODS))
            for base in ('bm25', 'bge'):
                initial = records[0]['methods'][base]['hits']
                reranked = records[0]['methods'][base + '_rerank']['hits']
                self.assertEqual({h['doc_id'] for h in initial}, {h['doc_id'] for h in reranked})
                initial_scores = {h['doc_id']: h['score'] for h in initial}
                self.assertTrue(all(h['retrieval_score'] == initial_scores[h['doc_id']] for h in reranked))
            # A legacy result must migrate without rerunning either retriever.
            legacy = read_json(run / 'metrics.json')
            for method in METHODS:
                legacy['aggregate'][method].pop('precision@5')
                legacy['aggregate'][method]['subset_recall@1'] = {'mean': 1, 'valid_queries': 1}
                legacy['per_query'][method]['sample-1'].pop('precision@5')
                legacy['per_query'][method]['sample-1']['subset_recall@1'] = 1
            write_json(run / 'metrics.json', legacy)
            ranking_hash = digest(run / 'retrieval.jsonl')
            report = generate_report(run, verify_metrics=True)
            self.assertEqual(digest(run / 'retrieval.jsonl'), ranking_hash)
            refreshed = read_json(run / 'metrics.json')
            for method in METHODS:
                self.assertIn('precision@5', refreshed['aggregate'][method])
            for file in ('metrics.json', 'metrics.csv', 'summary.md', 'report/index.html'):
                text = (run / file).read_text(encoding='utf-8').lower()
                self.assertIn('precision@5', text)
                self.assertNotIn('subset_recall', text)
            self.assertTrue(read_json(run / 'metrics_validation.json')['passed'])
            self.assertEqual(read_json(run / 'metrics_validation.json')['checks']['precision'], 4)
            detail = next(report.parent.glob('q-*.html')).read_text(encoding='utf-8')
            self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', detail)
            self.assertNotIn('href="javascript:', detail)
            self.assertNotIn('Qwen 生成答案', detail)
            self.assertNotIn('尚未生成答案', detail)
            self.assertIn('检索评测记录', report.read_text(encoding='utf-8'))
            self.assertFalse((run / 'answers.jsonl').exists())



if __name__ == '__main__':
    unittest.main()
