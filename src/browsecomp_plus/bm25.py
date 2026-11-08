"""Read the official Lucene index using the benchmark's default BM25 searcher."""
from pathlib import Path

from ..schemas import Hit


class BM25Retriever:
    def __init__(self, index_dir: Path):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(index_dir))
        self.document_count = self.searcher.num_docs

    def retrieve(self, query: str, k: int) -> list[Hit]:
        results = []
        for rank, hit in enumerate(self.searcher.search(query, k), 1):
            score = float(hit.score)
            results.append(Hit(str(hit.docid), rank, score, rank, score))
        return results
