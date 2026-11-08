from pathlib import Path
from ..schemas import Hit
from ..utils import read_json, write_json


class BM25Retriever:
    def __init__(self, docs, config):
        import bm25s
        self.bm25s, self.docs, self.config = bm25s, docs, config
        self.stemmer = None
        if config['stem']:
            import Stemmer
            self.stemmer = Stemmer.Stemmer('english')
        self.index = None

    def tokenize(self, texts):
        return self.bm25s.tokenize(texts, stopwords='en', stemmer=self.stemmer, show_progress=False)

    def build(self, directory, fingerprint):
        self.index = self.bm25s.BM25(k1=self.config['k1'], b=self.config['b'], method='lucene')
        self.index.index(self.tokenize([doc.search_text for doc in self.docs]), show_progress=False)
        directory = Path(directory)
        self.index.save(str(directory))
        write_json(directory / 'metadata.json', {'fingerprint': fingerprint, 'documents': len(self.docs)})

    def load(self, directory, fingerprint):
        if read_json(Path(directory) / 'metadata.json')['fingerprint'] != fingerprint:
            raise ValueError('BM25 索引与语料/配置不一致，请重新 index。')
        self.index = self.bm25s.BM25.load(str(directory), mmap=True)

    def retrieve(self, query, k):
        ids, scores = self.index.retrieve(self.tokenize([query]), k=min(k, len(self.docs)), show_progress=False)
        pairs = sorted(zip(ids[0], scores[0]), key=lambda x: (-float(x[1]), self.docs[int(x[0])].doc_id))
        return [Hit(self.docs[int(i)].doc_id, rank, float(score), rank, float(score))
                for rank, (i, score) in enumerate(pairs, 1)]
