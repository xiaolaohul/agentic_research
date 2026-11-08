"""Read the downloaded benchmark and prepare the official query list."""
from pathlib import Path
from runpy import run_path

from ..schemas import Document, Query
from ..utils import read_json


def benchmark_revisions(data_dir: Path, packages=('benchmark', 'indexes/bm25')) -> dict[str, str]:
    manifest = read_json(data_dir / 'download-manifest.json')
    revisions = {}
    for package in packages:
        entries = [entry for entry in manifest if entry['Package'].replace('\\', '/') == package]
        versions = {entry['Revision'] for entry in entries}
        if len(versions) != 1:
            raise ValueError(f'Expected one official revision for {package}.')
        for entry in entries:
            file = data_dir / package / entry['Name']
            if file.stat().st_size != entry['Bytes']:
                raise ValueError(f'Downloaded file size does not match the manifest: {file}')
        revisions[package] = versions.pop()
    return revisions


def corpus_batches(corpus_dir: Path, columns: list[str], batch_size: int):
    from pyarrow.parquet import ParquetFile

    shards = sorted(corpus_dir.glob('train-*.parquet'))
    if len(shards) != 7:
        raise ValueError(f'Expected 7 corpus Parquet files, found {len(shards)}.')
    for shard in shards:
        with ParquetFile(shard) as parquet:
            yield from parquet.iter_batches(batch_size=batch_size, columns=columns)


def load_candidate_documents(corpus_dir: Path, doc_ids: set[str]) -> list[Document]:
    found = {}
    for batch in corpus_batches(corpus_dir, ['docid', 'text'], 256):
        ids, texts = batch.column('docid'), batch.column('text')
        for row, doc_id in enumerate(ids.to_pylist()):
            doc_id = str(doc_id)
            if doc_id in doc_ids:
                text = texts[row].as_py()
                if doc_id in found or not text or not text.strip():
                    raise ValueError(f'Invalid corpus document for candidate {doc_id}.')
                found[doc_id] = Document(doc_id, text)
        if len(found) == len(doc_ids):
            return list(found.values())
    missing = sorted(doc_ids - found.keys())
    raise ValueError(f'{len(missing)} candidates are missing from the corpus: {missing[:10]}')


def load_queries(benchmark_dir: Path, decoder_path: Path, cache_dir: Path) -> list[Query]:
    from datasets import load_dataset

    shards = sorted(benchmark_dir.glob('test-*.parquet'))
    if len(shards) != 6:
        raise ValueError(f'Expected 6 benchmark Parquet files, found {len(shards)}.')
    decoder = run_path(str(decoder_path))
    rows = load_dataset(
        'parquet',
        data_files={'test': [str(shard) for shard in shards]},
        split='test',
        streaming=True,
        columns=['query_id', 'query'],
        cache_dir=str(cache_dir),
    )
    queries = []
    for row in rows:
        qid = str(row['query_id']).strip()
        text = decoder['decrypt_string'](row['query'], decoder['DEFAULT_CANARY'])
        text = text.replace('\t', ' ').replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()
        if not qid.isdecimal() or not text:
            raise ValueError(f'Invalid benchmark query record: {qid!r}')
        queries.append(Query(qid, text))
    if len(queries) != 830 or len({query.qid for query in queries}) != 830:
        raise ValueError(f'Expected 830 benchmark queries with distinct IDs, got {len(queries)} records.')
    return queries


def write_queries(path: Path, queries: list[Query]) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as file:
        for query in queries:
            file.write(f'{query.qid}\t{query.text}\n')
