from dataclasses import asdict, dataclass


METHODS = ('bm25', 'bge', 'bm25_rerank', 'bge_rerank')
HYBRID_METHODS = ('bm25', 'bge', 'rrf', 'bm25_rerank', 'bge_rerank')
LABELS = {'bm25': 'BM25', 'bge': 'BGE', 'bm25_rerank': 'BM25 + 重排序',
          'bge_rerank': 'BGE + 重排序', 'rrf': 'BM25 + BGE（RRF）'}


def methods_for(records):
    """Infer supported methods from a run, leaving legacy four-method runs valid."""
    if not records:
        raise ValueError('实验没有问题记录。')
    present = set(records[0]['methods'])
    if not present or not present.issubset(LABELS):
        raise ValueError('实验包含未知或空的检索方案。')
    if any(set(row['methods']) != present for row in records):
        raise ValueError('不同问题的检索方案不一致。')
    return tuple(method for method in HYBRID_METHODS if method in present)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ''
    url: str = ''
    parent_id: str = ''

    @property
    def search_text(self):
        return f'{self.title}\n{self.text}'.strip()


@dataclass(frozen=True)
class Query:
    qid: str
    text: str


@dataclass
class Hit:
    doc_id: str
    rank: int
    score: float
    retrieval_rank: int
    retrieval_score: float
    rerank_score: float | None = None

    def to_dict(self):
        return asdict(self)
