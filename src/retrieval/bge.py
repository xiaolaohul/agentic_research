from pathlib import Path
import numpy as np
from ..schemas import Hit
from ..utils import read_json, write_json


class BGERetriever:
    def __init__(self, docs, config):
        import faiss
        self.faiss, self.docs, self.config = faiss, docs, config
        self.model, self.index = None, None

    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config['model'], revision=self.config['revision'],
                                             device=self.config['device'], trust_remote_code=False,
                                             model_kwargs=self.config.get('model_kwargs'))
            self.model.max_seq_length = self.config['max_length']
        return self.model

    def build(self, directory, fingerprint, *, document_batches=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.load_model()
        dimension = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
        self.index = self.faiss.IndexFlatIP(dimension)
        # Only retain a batch of embeddings at a time, rather than duplicate the full matrix.
        batch = self.config['batch_size']
        if document_batches is None:
            document_batches = (self.docs[start:start + batch] for start in range(0, len(self.docs), batch))
        for documents in document_batches:
            start = self.index.ntotal
            expected = self.docs[start:start + len(documents)]
            if [doc.doc_id for doc in documents] != [doc.doc_id for doc in expected]:
                raise ValueError('BGE document batch does not match the index document order.')
            vectors = model.encode([d.search_text for d in documents],
                                   batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
            self.index.add(np.ascontiguousarray(vectors, dtype='float32'))
        if self.index.ntotal != len(self.docs):
            raise ValueError('BGE index does not contain one vector per document.')
        self.faiss.write_index(self.index, str(directory / 'vectors.faiss'))
        revision = getattr(model[0].auto_model.config, '_commit_hash', None)
        write_json(directory / 'metadata.json', {'fingerprint': fingerprint, 'documents': len(self.docs),
                                                'resolved_revision': revision})

    def load(self, directory, fingerprint):
        meta = read_json(Path(directory) / 'metadata.json')
        if meta['fingerprint'] != fingerprint:
            raise ValueError('BGE 索引与语料/模型配置不一致，请重新 index。')
        # Query encoding must use exactly the same revision as document encoding.
        if meta.get('resolved_revision'):
            self.config = dict(self.config, revision=meta['resolved_revision'])
        self.index = self.faiss.read_index(str(Path(directory) / 'vectors.faiss'))
        if self.index.ntotal != len(self.docs):
            raise ValueError('向量索引条数与段落数不一致。')

    def retrieve(self, query, k):
        vectors = self.load_model().encode([self.config['query_instruction'] + query],
                                          normalize_embeddings=True, show_progress_bar=False)
        scores, ids = self.index.search(np.ascontiguousarray(vectors, dtype='float32'), min(k, len(self.docs)))
        pairs = sorted(zip(ids[0], scores[0]), key=lambda x: (-float(x[1]), self.docs[int(x[0])].doc_id))
        return [Hit(self.docs[int(i)].doc_id, rank, float(score), rank, float(score))
                for rank, (i, score) in enumerate(pairs, 1)]
