"""Run BrowseComp-Plus retrieval and reranking baselines."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from .bm25 import BM25Retriever
from .dataset import benchmark_revisions, load_queries, write_queries
from .evaluation import QRELS_FILES, load_judgments, score_rankings, write_run
from ..utils import ROOT, digest, read_json, utc_now, versions, write_json


BASELINE_NAMES = {'bm25': 'bm25-baseline', 'bge': 'bge-m3-4096-baseline',
                  'bm25_rerank': 'bm25-rerank-512-baseline', 'bge_rerank': 'bge-rerank-512-baseline'}


def configure_java() -> str:
    java_home = ROOT / 'tools' / 'java21'
    java_bin = java_home / 'bin'
    if (java_bin / ('java.exe' if os.name == 'nt' else 'java')).is_file():
        os.environ['JAVA_HOME'] = str(java_home)
        os.environ['PATH'] = str(java_bin) + os.pathsep + os.environ.get('PATH', '')
    version_result = subprocess.run(['java', '-version'], capture_output=True, text=True, check=True, timeout=20)
    return version_result.stderr.splitlines()[0]


def run_baseline(output: Path, *, method: str = 'bm25', index_dir: Path | None = None,
                 device: str = 'cpu', query_limit: int | None = None,
                 reranker_url: str | None = None, source_run: Path | None = None) -> None:
    base_method, reranking, _ = method.partition('_rerank')
    if not reranking and (reranker_url is not None or source_run is not None):
        raise ValueError('--reranker-url and --source-run require a reranking method.')
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    started_at = utc_now()
    data_dir = ROOT / 'data' / 'browsecomp-plus'
    official_dir = ROOT / 'tools' / 'BrowseComp-Plus'
    decoder = official_dir / 'scripts_build_index' / 'decrypt_dataset.py'
    qrels_dir = official_dir / 'topics-qrels'
    depth = 100
    comparison = None

    if reranking:
        from .reranking import compare_rankings, load_saved_candidates, rerank_candidates, write_rank_changes

        if index_dir is not None:
            raise ValueError('Reranking reads saved candidates. Select them with --source-run.')
        revisions = benchmark_revisions(data_dir, ('benchmark', 'corpus'))
        description = f'Saved {base_method.upper()} results reranked with BGE reranker v2 M3'
        software = versions()
        ranker = 'BGE reranker'
    elif method == 'bm25':
        java = configure_java()
        revisions = benchmark_revisions(data_dir)
        index_dir = index_dir or data_dir / 'indexes' / 'bm25'
        retriever = BM25Retriever(index_dir)
        document_count = retriever.document_count
        description = 'Pyserini LuceneSearcher with BrowseComp-Plus default BM25 settings'
        method_settings = {}
        software = {'java': java, 'pyserini': version('pyserini')}
        ranker = 'Lucene'
    else:
        from .bge import INDEX_DIR, load_retriever

        revisions = benchmark_revisions(data_dir, ('benchmark', 'corpus'))
        index_dir = index_dir or INDEX_DIR
        retriever = load_retriever(index_dir, revisions['corpus'], device, require_full=query_limit is None)
        document_count = len(retriever.docs)
        description = 'BGE-M3 dense retrieval with normalized vectors and exact FAISS inner product'
        method_settings = {'index_build': read_json(index_dir / 'build.json'), 'query_device': device}
        software = versions()
        ranker = 'FAISS'

    queries = load_queries(data_dir / 'benchmark', decoder, ROOT / '.cache' / 'browsecomp-datasets')
    judgments = load_judgments(qrels_dir, {query.qid for query in queries})
    if query_limit is not None:
        if not 1 <= query_limit < len(queries):
            raise ValueError(f'Choose a query limit between 1 and {len(queries) - 1}.')
        queries = queries[:query_limit]
    if reranking:
        source = source_run or ROOT / 'results' / 'browsecomp-plus' / BASELINE_NAMES[base_method]
        read_start = perf_counter()
        candidates, source_metadata = load_saved_candidates(
            source, queries, revisions, depth, base_method, require_full=query_limit is None)
        read_seconds = perf_counter() - read_start
        document_count = source_metadata['index_document_count']
        index_dir = ROOT / source_metadata['index']
        print(f'Replaying {len(queries)} questions with {min(depth, document_count)} saved '
              f'{base_method.upper()} hits each.', flush=True)
        rankings, config, times, timing = rerank_candidates(
            data_dir / 'corpus', queries, candidates, device, depth, reranker_url=reranker_url)
        timing['candidate_read_seconds'] = read_seconds
        method_settings = {'source_run': (source.relative_to(ROOT) if source.is_relative_to(ROOT) else source).as_posix(),
                           'source_run_sha256': source_metadata['outputs_sha256'][f'{base_method}.run'],
                           'reranker': config}
    else:
        print(f'Loaded {len(queries)} queries and both qrels sets for {document_count:,} documents.', flush=True)
        search_start = perf_counter()
        rankings = {}
        for number, query in enumerate(queries, 1):
            rankings[query.qid] = retriever.retrieve(query.text, depth)
            if number == 1 or number % 100 == 0 or number == len(queries):
                print(f'{method.upper()}: {number}/{len(queries)} queries searched', flush=True)
        timing = {'retrieval_seconds': perf_counter() - search_start}

    output.parent.mkdir(parents=True, exist_ok=True)
    run_file = f'{method}.run'
    with TemporaryDirectory(prefix=f'.{method}-', dir=output.parent) as directory:
        stage = Path(directory).resolve()
        if stage.parent != output.parent:
            raise ValueError('Temporary output escaped the intended directory.')
        write_queries(stage / 'queries.tsv', queries)
        write_run(stage / run_file, rankings, method, depth)
        metrics = {name: score_rankings(rankings, labels) for name, labels in judgments.items()}
        write_json(stage / 'metrics.json', metrics)
        artifacts = ['queries.tsv', run_file, 'metrics.json']
        if reranking:
            comparison = compare_rankings(candidates, metrics, judgments, times, base_method)
            write_json(stage / 'comparison.json', comparison)
            write_rank_changes(stage / 'rank_changes.tsv', rankings, base_method)
            artifacts.extend(('comparison.json', 'rank_changes.tsv'))
        write_json(stage / 'run.json', {
            'method': method,
            'retriever': description,
            'index': (index_dir.relative_to(ROOT) if index_dir.is_relative_to(ROOT) else index_dir).as_posix(),
            **method_settings,
            'download_revisions': revisions,
            'download_manifest_sha256': digest(data_dir / 'download-manifest.json'),
            'decoder_sha256': digest(decoder),
            'qrels': {name: {'file': filename, 'sha256': digest(qrels_dir / filename)}
                      for name, filename in QRELS_FILES.items()},
            'software': {'python': platform.python_version(), **software,
                         **{name: version(name) for name in ('datasets', 'pytrec-eval-terrier')}},
            'started_at': started_at,
            'finished_at': utc_now(),
            **timing,
            'query_count': len(queries),
            'index_document_count': document_count,
            'max_hits_per_query': depth,
            'returned_documents': sum(map(len, rankings.values())),
            'queries_without_hits': [qid for qid, hits in rankings.items() if not hits],
            'queries_with_score_ties': sum(len({hit.score for hit in hits}) < len(hits)
                                           for hits in rankings.values()),
            'run_scores': f'Original {method.upper()} scores; rank column records the {ranker} result order.',
            'metric_order': f'{ranker} rank, supplied to trec_eval as reciprocal ranks to preserve ties.',
            'outputs_sha256': {name: digest(stage / name) for name in artifacts},
        })
        stage.rename(output)

    for name, report in metrics.items():
        print(f'{name}: {report["overall"] if comparison is None else comparison[name]}', flush=True)
    print(f'Results: {output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(BASELINE_NAMES), default='bm25')
    parser.add_argument('--index-dir', type=Path)
    parser.add_argument('--source-run', type=Path, help='Saved retrieval result directory for reranking.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cpu', help='Device for BGE query encoding or reranking.')
    parser.add_argument('--query-limit', type=int, help='Use the first N queries for a smoke run.')
    parser.add_argument('--reranker-url', help='Local HTTP address of the AutoDL reranker SSH tunnel.')
    args = parser.parse_args()
    if not args.method.endswith('_rerank') and (args.reranker_url is not None or args.source_run is not None):
        parser.error('--reranker-url and --source-run require a reranking method')
    output = args.output_dir
    if output is None:
        name = BASELINE_NAMES[args.method]
        if args.reranker_url is not None:
            name = name.replace('-baseline', '-autodl-baseline')
        if args.query_limit is not None:
            name = name.replace('-baseline', f'-smoke-{args.query_limit}')
        output = ROOT / 'results' / 'browsecomp-plus' / name
    index = (ROOT / args.index_dir).resolve() if args.index_dir is not None else None
    source = (ROOT / args.source_run).resolve() if args.source_run is not None else None
    run_baseline(ROOT / output, method=args.method, index_dir=index, device=args.device,
                 query_limit=args.query_limit, reranker_url=args.reranker_url, source_run=source)


if __name__ == '__main__':
    main()
