"""Adapt the existing BrowseComp-Plus retrievers to the Tongyi search tool."""

import json
from functools import lru_cache
from time import perf_counter

from .reranking import request_remote_ranking


class AgentSearch:
    def __init__(self, retriever, lucene_index, *, reranker_url=None, reranker=None):
        self.retriever = retriever
        self.lucene_index = lucene_index
        self.reranker_url = reranker_url
        self.reranker = reranker

    @lru_cache(maxsize=256)
    def document_text(self, docid):
        document = self.lucene_index.doc(docid)
        if document is None:
            raise ValueError(f'Document {docid} is missing from the official Lucene index.')
        return json.loads(document.raw())['contents']

    def search(self, query, k):
        return self.search_with_details(query, k)[0]

    def search_with_details(self, query, k):
        started = perf_counter()
        depth = self.reranker['config']['depth'] if self.reranker else k
        candidates = self.retriever.retrieve(query, depth)
        details = {'retrieval_seconds': perf_counter() - started}
        ranking = candidates

        if self.reranker and candidates:
            started = perf_counter()
            texts = {hit.doc_id: self.document_text(hit.doc_id) for hit in candidates}
            details['document_lookup_seconds'] = perf_counter() - started
            started = perf_counter()
            ranking, request_seconds, model_seconds, software, _ = request_remote_ranking(
                self.reranker_url, query, candidates, texts, self.reranker['config'])
            if software != self.reranker['software']:
                raise ValueError('The remote reranker software changed after preflight.')
            details.update(
                rerank_seconds=perf_counter() - started,
                rerank_request_seconds=request_seconds,
                rerank_model_seconds=model_seconds,
                candidates=[hit.to_dict() for hit in sorted(ranking, key=lambda hit: hit.retrieval_rank)],
            )
        elif self.reranker:
            details['candidates'] = []

        documents = [
            {**hit.to_dict(), 'docid': hit.doc_id, 'text': self.document_text(hit.doc_id)}
            for hit in ranking[:k]
        ]
        return documents, details
