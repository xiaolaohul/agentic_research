"""Build and load the BrowseComp-Plus BGE-M3 document index."""
import argparse
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .dataset import benchmark_revisions, corpus_batches
from ..retrieval.bge import BGERetriever
from ..schemas import Document
from ..utils import ROOT, object_hash, read_json, versions, write_json


MODEL = 'BAAI/bge-m3'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
MAX_LENGTH = 4096
DOCUMENT_COUNT = 100_195
DATA_DIR = ROOT / 'data' / 'browsecomp-plus'
INDEX_DIR = DATA_DIR / 'indexes' / 'bge-m3-4096'
MODEL_DIR = ROOT / 'models' / 'huggingface' / 'hub' / 'models--BAAI--bge-m3' / 'snapshots' / REVISION


def model_config(device: str, batch_size: int = 1) -> dict:
    if not (MODEL_DIR / 'pytorch_model.bin').is_file():
        raise FileNotFoundError(f'Install the pinned {MODEL} snapshot at {MODEL_DIR}.')
    return {
        'model': str(MODEL_DIR),
        'revision': REVISION,
        'device': device,
        'batch_size': batch_size,
        'max_length': MAX_LENGTH,
        'query_instruction': '',
        'model_kwargs': {'torch_dtype': 'float16'} if device.startswith('cuda') else None,
    }


def _fingerprint(corpus_revision: str, doc_ids: list[str]) -> str:
    return object_hash({
        'model': MODEL, 'revision': REVISION, 'max_length': MAX_LENGTH,
        'corpus_revision': corpus_revision, 'docids': doc_ids,
    })


def _validate_docids(doc_ids: list[str], expected_count: int) -> None:
    if (len(doc_ids) != expected_count or len(set(doc_ids)) != expected_count
            or any(not isinstance(doc_id, str) or not doc_id.isdecimal() for doc_id in doc_ids)):
        raise ValueError(f'Expected {expected_count:,} distinct official corpus docids.')


def _documents(corpus_dir: Path, count: int, batch_size: int):
    processed = 0
    for batch in corpus_batches(corpus_dir, ['docid', 'text'], batch_size):
        rows = batch.slice(0, count - processed).to_pylist()
        yield [Document(str(row['docid']), row['text']) for row in rows]
        previous = processed
        processed += len(rows)
        if processed // 1000 > previous // 1000 or processed == count:
            print(f'Encoded {processed:,}/{count:,} documents.', flush=True)
        if processed == count:
            return


def build_index(index_dir: Path, device: str, batch_size: int, document_limit: int | None = None) -> None:
    if index_dir.exists():
        raise FileExistsError(f'Index output already exists: {index_dir}')
    if batch_size < 1 or (document_limit is not None and not 1 <= document_limit < DOCUMENT_COUNT):
        raise ValueError('Choose a positive batch size and a document limit below the full corpus size.')
    if document_limit is not None and index_dir == INDEX_DIR:
        raise ValueError('A sample index must use a separate output directory.')

    corpus_revision = benchmark_revisions(DATA_DIR, ('corpus',))['corpus']
    corpus_dir = DATA_DIR / 'corpus'
    expected = DOCUMENT_COUNT if document_limit is None else document_limit
    ids = (str(doc_id) for batch in corpus_batches(corpus_dir, ['docid'], 4096)
           for doc_id in batch.column(0).to_pylist())
    doc_ids = list(islice(ids, document_limit))
    ids.close()
    _validate_docids(doc_ids, expected)
    retriever = BGERetriever([Document(doc_id, '') for doc_id in doc_ids], model_config(device, batch_size))

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with TemporaryDirectory(prefix='.bge-', dir=index_dir.parent) as temp:
        stage = Path(temp) / 'index'
        retriever.build(stage, _fingerprint(corpus_revision, doc_ids),
                        document_batches=_documents(corpus_dir, expected, batch_size))
        write_json(stage / 'docids.json', doc_ids)
        write_json(stage / 'build.json', {
            'model': MODEL, 'model_revision': REVISION, 'max_length': MAX_LENGTH,
            'corpus_revision': corpus_revision, 'documents': expected, 'device': device,
            'precision': 'float16' if device.startswith('cuda') else 'float32',
            'batch_size': batch_size, 'encoding_seconds': perf_counter() - started,
            'software': versions(),
        })
        stage.rename(index_dir)
    print(f'Index: {index_dir}', flush=True)


def load_retriever(index_dir: Path, corpus_revision: str, device: str, *, require_full: bool = True) -> BGERetriever:
    doc_ids = read_json(index_dir / 'docids.json')
    if not doc_ids:
        raise ValueError('The BGE index is empty.')
    _validate_docids(doc_ids, DOCUMENT_COUNT if require_full else len(doc_ids))
    retriever = BGERetriever([Document(doc_id, '') for doc_id in doc_ids], model_config(device))
    retriever.load(index_dir, _fingerprint(corpus_revision, doc_ids))
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--document-limit', type=int)
    args = parser.parse_args()
    index_dir = args.index_dir
    if index_dir is None:
        index_dir = INDEX_DIR if args.document_limit is None else INDEX_DIR.with_name(
            f'{INDEX_DIR.name}-smoke-{args.document_limit}')
    build_index((ROOT / index_dir).resolve(), args.device, args.batch_size, args.document_limit)


if __name__ == '__main__':
    main()
