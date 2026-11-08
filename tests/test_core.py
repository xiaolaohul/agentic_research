import copy
from dataclasses import asdict
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.dataset import load_documents, load_qrels, normalize_document, prepare, save_prepared, validate_prepared
from src.evaluation import evaluate_query
from src.schemas import Document, Query, Hit
from src.utils import ROOT, read_json, write_jsonl, atomic_text


def temporary_directory():
    root = ROOT / '.cache/test-tmp'
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class MetricsTests(unittest.TestCase):
    def test_corpus_ndcg_perfect_sort_ignores_outside_positives(self):
        labels = {'a': 2, 'b': 1, 'outside1': 3, 'outside2': 3}
        hits = [{'doc_id': 'a'}, {'doc_id': 'b'}]
        local = evaluate_query(hits, labels, {'a', 'b'}, [2], ndcg_scope='corpus')
        original = evaluate_query(hits, labels, {'a', 'b'}, [2], ndcg_scope='all_judgments')
        self.assertEqual(local['ndcg@2'], 1.0)
        self.assertLess(original['ndcg@2'], 1.0)
        self.assertEqual(local['judged@2'], original['judged@2'])

    def test_corpus_ndcg_uses_unretrieved_positive_in_shared_corpus(self):
        labels = {'a': 3, 'b': 1, 'outside': 3}
        result = evaluate_query([{'doc_id': 'b'}, {'doc_id': 'unknown'}], labels,
                                {'a', 'b', 'unknown'}, [2], ndcg_scope='corpus')
        self.assertAlmostEqual(result['ndcg@2'], 1 / (3 + 1 / math.log2(3)))

    def test_corpus_ndcg_zero_grade_and_no_local_positive(self):
        result = evaluate_query([{'doc_id': 'zero'}], {'zero': 0, 'outside': 3},
                                {'zero'}, [10], ndcg_scope='corpus')
        self.assertIsNone(result['ndcg@10'])
        self.assertEqual(result['judged@10'], .1)
        with self.assertRaises(ValueError):
            evaluate_query([], {}, set(), [10], ndcg_scope='invalid')

    def test_independent_validation_corpus_scope_and_short_results(self):
        from src.evaluation import evaluate
        from src.validation import validate_metrics
        from src.schemas import METHODS
        from src.utils import write_json
        with temporary_directory() as folder:
            run = Path(folder)
            write_jsonl(run / 'corpus.jsonl', [asdict(Document(d, d)) for d in ['a', 'b', 'zero']])
            atomic_text(run / 'qrels.tsv', 'q 0 a 3\nq 0 b 1\nq 0 zero 0\nq 0 outside 3\n')
            records = [{'qid': 'q', 'methods': {m: {'hits': [{'doc_id': 'b', 'rank': 1}],
                        'retrieval_ms': 1, 'rerank_ms': 0, 'total_ms': 1} for m in METHODS}}]
            write_jsonl(run / 'retrieval.jsonl', records)
            evaluation = {'ks': [2, 5], 'gain': 'linear', 'relevant_threshold': 1, 'ndcg_scope': 'corpus'}
            write_json(run / 'config.json', {'evaluation': evaluation})
            metrics = evaluate(records, load_qrels(run / 'qrels.tsv'), {'a', 'b', 'zero'}, evaluation)
            write_json(run / 'metrics.json', metrics)
            self.assertTrue(validate_metrics(run)['passed'])
            # A validator that accidentally uses all original positives must reject these scores.
            write_json(run / 'config.json', {'evaluation': {**evaluation, 'ndcg_scope': 'all_judgments'}})
            with self.assertRaisesRegex(ValueError, '独立指标核验失败'):
                validate_metrics(run)

    def test_full_denominator_and_linear_ndcg(self):
        hits = [{'doc_id': 'b'}, {'doc_id': 'unjudged'}, {'doc_id': 'a'}]
        labels = {'a': 3, 'b': 1, 'outside': 2, 'irrelevant': 0}
        result = evaluate_query(hits, labels, {'a', 'b', 'unjudged'}, [2, 3])
        self.assertAlmostEqual(result['recall@2'], 1 / 3)
        self.assertAlmostEqual(result['precision@5'], 2 / 5)
        self.assertFalse(any(key.startswith('subset_recall') for key in result))
        self.assertAlmostEqual(result['corpus_relevant_coverage'], 2 / 3)
        expected = (1 + 3 / math.log2(4)) / (3 + 2 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(result['ndcg@3'], expected)
        self.assertEqual(result['mrr@3'], 1.0)
        self.assertEqual(result['judged@2'], 0.5)

    def test_missing_labels_not_fabricated_zero(self):
        result = evaluate_query([{'doc_id': 'a'}], {}, {'a'}, [5])
        self.assertIsNone(result['ndcg@5'])
        self.assertIsNone(result['recall@5'])
        self.assertIsNone(result['mrr@5'])
        self.assertIsNone(result['precision@5'])
        self.assertFalse(result['judged_query'])

    def test_precision5_cutoff_grades_and_missing_slots(self):
        hits = [{'doc_id': doc} for doc in ['a', 'b', 'c', 'zero', 'unknown', 'sixth']]
        labels = {'a': 1, 'b': 2, 'c': 3, 'zero': 0, 'sixth': 3}
        self.assertAlmostEqual(evaluate_query(hits, labels, set(labels), [10])['precision@5'], 3 / 5)
        self.assertAlmostEqual(evaluate_query(hits, labels, set(labels), [10], threshold=2)['precision@5'], 2 / 5)
        self.assertAlmostEqual(evaluate_query(hits[:1], labels, set(labels), [10])['precision@5'], 1 / 5)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_query([{'doc_id': 'a'}, {'doc_id': 'a'}], {'a': 1}, {'a'}, [5])


class DatasetTests(unittest.TestCase):
    def config(self, root):
        return {'dataset': {'name': 'fixture', 'corpus': str(root / 'corpus.jsonl'),
                'queries': str(root / 'queries.jsonl'), 'qrels': str(root / 'qrels.tsv'),
                'manifest': str(root / 'manifest.json'), 'seed': 42, 'max_documents': 10,
                'max_queries': 2, 'judgments': 'UMBRELA test fixture', 'raw_dir': str(root)},
                'storage': {'reserve_gib': 0, 'budget_gib': 100000}}

    def test_segment_mapping_and_fingerprint(self):
        with temporary_directory() as folder:
            root, doc_id = Path(folder), 'msmarco_v2.1_doc_00_100#0_20'
            cfg = self.config(root)
            doc = normalize_document({'docid': doc_id, 'doc': {'segment': 'passage', 'title': 'Title'}})
            self.assertEqual(doc.parent_id, 'msmarco_v2.1_doc_00_100')
            save_prepared(cfg, [doc], [Query('2024-1', 'question')],
                          {'2024-1': {doc_id: 2, 'outside': 1}}, {'kind': 'corpus_sample'})
            _, _, labels, _ = validate_prepared(cfg)
            self.assertIn('outside', labels['2024-1'])
            atomic_text(root / 'corpus.jsonl', '{}\n')
            with self.assertRaisesRegex(ValueError, 'manifest'):
                validate_prepared(cfg)

    def test_parent_doc_qrels_rejected(self):
        with temporary_directory() as folder:
            cfg = self.config(Path(folder))
            doc = Document('msmarco_v2.1_doc_00_100#0_20', 'text')
            with self.assertRaisesRegex(ValueError, 'ID'):
                save_prepared(cfg, [doc], [Query('2024-1', 'question')],
                    {'2024-1': {'msmarco_v2.1_doc_00_100': 2}}, {'kind': 'corpus_sample'})

    def test_pool_dedup_and_order_independent_sampling(self):
        with temporary_directory() as folder:
            root = Path(folder)
            cfg = self.config(root)
            doc_id = 'msmarco_v2.1_doc_00_100#0_20'
            candidate = {'docid': doc_id, 'doc': {'segment': 'passage'}}
            pool = [{'query': {'qid': q}, 'candidates': [candidate]} for q in ('2024-1', '2024-2')]
            write_jsonl(root / 'pool.jsonl', pool)
            atomic_text(root / 'topics.txt', '2024-1\tquestion one\n2024-2\tquestion two\n')
            atomic_text(root / 'labels.tsv', f'2024-1 0 {doc_id} 2\n')
            first = prepare(cfg, root / 'pool.jsonl', str(root / 'topics.txt'), str(root / 'labels.tsv'), pool=True)
            self.assertEqual(first['documents'], 1)
            write_jsonl(root / 'pool.jsonl', reversed(pool))
            second = prepare(cfg, root / 'pool.jsonl', str(root / 'topics.txt'), str(root / 'labels.tsv'), pool=True, overwrite=True)
            self.assertEqual(first['corpus_sha256'], second['corpus_sha256'])
            self.assertEqual(first['query_ids'], second['query_ids'])

    def test_unjudged_negative_qrels(self):
        with temporary_directory() as folder:
            file = Path(folder) / 'qrels'
            atomic_text(file, 'q 0 unjudged -1\nq 0 irrelevant 0\nq 0 relevant 2\n')
            self.assertEqual(load_qrels(file), {'q': {'irrelevant': 0, 'relevant': 2}})

    def test_global_pool_is_shared_and_does_not_use_qrels_for_selection(self):
        with temporary_directory() as folder:
            root = Path(folder)
            cfg = self.config(root)
            cfg['dataset']['max_queries'] = 1
            cfg['dataset']['pool_scope'] = 'all'
            pool, labels = [], []
            for i in range(1, 4):
                doc = f'msmarco_v2.1_doc_00_{i}#0_{i}'
                pool.append({'query': {'qid': f'2024-{i}', 'text': f'question {i}'},
                             'candidates': [{'docid': doc, 'doc': {'segment': f'passage {i}'}}]})
                labels.append(f'2024-{i} 0 {doc} 2\n')
            write_jsonl(root / 'pool.jsonl', pool)
            atomic_text(root / 'topics.txt', ''.join(f'2024-{i}\tquestion {i}\n' for i in range(1, 4)))
            atomic_text(root / 'labels.tsv', ''.join(labels))
            first = prepare(cfg, root / 'pool.jsonl', str(root / 'topics.txt'), str(root / 'labels.tsv'), pool=True)
            self.assertEqual(first['queries'], 1)
            self.assertEqual(first['documents'], 3)
            self.assertEqual(first['provenance']['pool_available_queries'], 3)
            atomic_text(root / 'labels.tsv', ''.join(labels).replace(' 2\n', ' 0\n'))
            second = prepare(cfg, root / 'pool.jsonl', str(root / 'topics.txt'), str(root / 'labels.tsv'), pool=True, overwrite=True)
            self.assertEqual(first['corpus_sha256'], second['corpus_sha256'])
            self.assertEqual(first['query_ids'], second['query_ids'])

    def test_pool_cannot_relabel_an_official_query_id(self):
        with temporary_directory() as folder:
            root = Path(folder)
            cfg = self.config(root)
            write_jsonl(root / 'pool.jsonl', [{'query': {'qid': '2024-1', 'text': 'wrong question'}, 'candidates': []}])
            atomic_text(root / 'topics.txt', '2024-1\tofficial question\n')
            with self.assertRaisesRegex(ValueError, 'topics'):
                prepare(cfg, root / 'pool.jsonl', str(root / 'topics.txt'), str(root / 'missing.tsv'), pool=True)


class BM25Tests(unittest.TestCase):
    def test_saved_index_preserves_document_mapping(self):
        from src.retrieval.bm25 import BM25Retriever
        docs = [Document('cat', 'A cat purrs.'), Document('wind', 'Wind turbines generate electricity.')]
        config = {'stem': True, 'k1': 0.9, 'b': 0.4}
        with temporary_directory() as folder:
            first = BM25Retriever(docs, config)
            first.build(folder, 'fixture')
            second = BM25Retriever(docs, config)
            second.load(folder, 'fixture')
            self.assertEqual(second.retrieve('turbine electricity', 100)[0].doc_id, 'wind')
            self.assertEqual(len(second.retrieve('turbine electricity', 100)), 2)
            with self.assertRaises(ValueError):
                second.load(folder, 'wrong-corpus')
            del first, second
            import gc
            gc.collect()


class DownloadTests(unittest.TestCase):
    def test_interrupted_download_can_retry_and_resume(self):
        from main import retry_download
        from requests.exceptions import ChunkedEncodingError
        from unittest.mock import Mock
        action = Mock(side_effect=[ChunkedEncodingError('interrupted'), 'complete-cache-path'])
        with patch('main.time.sleep'):
            self.assertEqual(retry_download(action), 'complete-cache-path')
        self.assertEqual(action.call_count, 2)

    def test_access_error_is_not_retried(self):
        from main import retry_download
        from requests import Response
        from requests.exceptions import HTTPError
        from unittest.mock import Mock
        response = Response()
        response.status_code = 403
        action = Mock(side_effect=HTTPError(response=response))
        with self.assertRaises(ValueError):
            retry_download(action)
        self.assertEqual(action.call_count, 1)


class AtomicWriteTests(unittest.TestCase):
    def test_transient_windows_file_lock_does_not_lose_checkpoint(self):
        with temporary_directory() as folder:
            file = Path(folder) / 'checkpoint.json'
            atomic_text(file, 'old')
            original, calls = Path.replace, []
            def replace(source, target):
                calls.append(str(source))
                if len(calls) == 1:
                    raise PermissionError('temporary Windows file sharing lock')
                return original(source, target)
            with patch('pathlib.Path.replace', replace), patch('src.utils.time.sleep'):
                atomic_text(file, 'new')
            self.assertEqual(file.read_text(encoding='utf-8'), 'new')
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_permanent_access_error_preserves_previous_checkpoint(self):
        with temporary_directory() as folder:
            file = Path(folder) / 'checkpoint.json'
            atomic_text(file, 'previous')
            with patch('pathlib.Path.replace', side_effect=PermissionError('locked')), patch('src.utils.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_text(file, 'replacement')
            self.assertEqual(file.read_text(encoding='utf-8'), 'previous')


if __name__ == '__main__':
    unittest.main()
