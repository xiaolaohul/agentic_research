"""Run the four BrowseComp-Plus search tools with the same Tongyi agent."""

import argparse
from datetime import date
import json
import os
from pathlib import Path
from random import Random
import sys
from urllib.request import ProxyHandler, build_opener

from .dataset import benchmark_revisions, write_queries
from .evaluation import load_judgments
from .run_retrieval import configure_java
from ..schemas import Query
from ..utils import ROOT, digest, load_config, read_json, utc_now, write_json

DATA = ROOT / 'data' / 'browsecomp-plus'
OFFICIAL = ROOT / 'tools' / 'BrowseComp-Plus'
RESULTS = ROOT / 'results' / 'browsecomp-plus'
QUERY_SOURCE = RESULTS / 'bm25-baseline' / 'queries.tsv'
METHODS = ('bm25', 'bm25_rerank', 'bge', 'bge_rerank')
PILOT_IDS = ('769', '770', '771', '772', '773')
STRATA = ((1, 4, 8), (5, 7, 13), (8, 100_195, 9))
SAMPLE_SEED = 20261109
MODEL = 'Alibaba-NLP/Tongyi-DeepResearch-30B-A3B'
DEPTH = 100
CONTEXT = 32768
SNIPPET_TOKENS = 512
TOP_K = 5
GENERATION = {'temperature': 0.85, 'top_p': 0.95, 'presence_penalty': 1.1,
              'max_output_tokens': 4096, 'max_retries': 2, 'context_size': CONTEXT}
SERVICE_ERRORS = {'model server error', 'search service error'}


def load_questions():
    rows = [Query(*line.split('\t', 1)) for line in QUERY_SOURCE.read_text(encoding='utf-8').splitlines()]
    questions = {q.qid: q for q in rows}
    if len(rows) != 830 or len(questions) != len(rows) or any(not q.text or not q.qid.isdecimal() for q in rows):
        raise ValueError('Expected 830 distinct benchmark questions.')
    labels = load_judgments(OFFICIAL / 'topics-qrels', set(questions))['evidence']
    return questions, labels


def select_questions(questions, evidence):
    rng = Random(SAMPLE_SEED)
    eligible = sorted(questions.keys() - set(PILOT_IDS), key=int)
    selected = []
    for lower, upper, count in STRATA:
        pool = [qid for qid in eligible if lower <= len(evidence[qid]) <= upper]
        selected.extend(rng.sample(pool, count))
    rng.shuffle(selected)
    return selected


def endpoint_json(url):
    with build_opener(ProxyHandler({})).open(url, timeout=8) as response:
        return json.load(response)


def preflight(tongyi_port, reranker_port, needs_reranker):
    tongyi = f'http://127.0.0.1:{tongyi_port}'
    if endpoint_json(f'{tongyi}/health').get('status') != 'ok':
        raise RuntimeError('Tongyi is not ready.')
    models = endpoint_json(f'{tongyi}/v1/models')['data']
    if MODEL not in {entry['id'] for entry in models}:
        raise ValueError('The Tongyi alias does not match the experiment model.')
    props = endpoint_json(f'{tongyi}/props')
    if props['total_slots'] != 1 or props['default_generation_settings']['n_ctx'] != CONTEXT:
        raise ValueError('Expected Tongyi with one slot and 32768 context tokens.')
    reranker_url = f'http://127.0.0.1:{reranker_port}'
    reranker = None
    if needs_reranker:
        reranker = endpoint_json(f'{reranker_url}/health')
        config = load_config(ROOT / 'configs' / 'default.yaml')
        expected = dict(config['reranker'], depth=DEPTH, device='cuda', max_length=512)
        if reranker.get('config') != expected or not reranker.get('software', {}).get('gpu'):
            raise ValueError('The remote reranker configuration or GPU differs from the experiment.')
    return {'model_path': props['model_path'], 'total_slots': props['total_slots']}, reranker_url, reranker


def run_path(output, qid, method):
    return output / 'runs' / qid / f'{method}.json'


def metric_fields(result, evidence):
    searches = result['searches']
    delivered = [search for search in searches if search['delivered'] and search['docids']]
    seen = set()
    first_hit = None
    for search in delivered:
        seen.update(search['docids'])
        hits = sorted(set(search['docids']) & evidence, key=int)
        search['evidence_hits'] = hits
        if hits and first_hit is None:
            first_hit = search['round']
    matched = sorted(seen & evidence, key=int)
    usage = {}
    for key in ('prompt_tokens', 'completion_tokens'):
        counts = [call[key] for call in result['model_calls']]
        usage[key] = sum(counts) if counts and None not in counts else None
    usage['total_tokens'] = sum(usage.values()) if None not in usage.values() else None
    return {
        'search_attempts': len(searches),
        'valid_searches': sum(search['docids'] is not None for search in searches),
        'unique_delivered_documents': len(seen),
        'evidence_docids': sorted(evidence, key=int),
        'matched_evidence_docids': matched,
        'evidence_recall': len(matched) / len(evidence),
        'first_evidence_round': first_hit,
        'token_usage': usage,
        'retrieval_seconds': sum(s.get('retrieval_seconds', 0) for s in searches),
        'rerank_seconds': sum(s.get('rerank_seconds', 0) for s in searches),
        'model_seconds': sum(call['elapsed_seconds'] for call in result['model_calls']),
    }


def experiment_settings(output, args, revisions, server, reranker, max_calls):
    files = [
        Path(__file__), Path(__file__).with_name('agent_search.py'),
        OFFICIAL / 'search_agent/tongyi_utils/react_agent.py',
        OFFICIAL / 'search_agent/tongyi_utils/tool_search.py',
        DATA / 'indexes/bge-m3-4096/build.json',
    ]
    frozen = output / 'manifest.json'
    previous = read_json(frozen) if frozen.is_file() else None
    return {
        'model': MODEL, 'server': server, 'reranker': reranker, 'revisions': revisions,
        'prompt_date': previous['settings']['prompt_date'] if previous else date.today().isoformat(),
        'top_k': TOP_K, 'snippet_max_tokens': SNIPPET_TOKENS, 'rerank_depth': DEPTH,
        'max_llm_calls': max_calls, 'generation': GENERATION, 'bge_query_device': 'cpu',
        'model_seed_rule': f'{SAMPLE_SEED} + numeric question ID',
        'source_sha256': digest(QUERY_SOURCE), 'evidence_sha256': digest(OFFICIAL / 'topics-qrels/qrel_evidence.txt'),
        'implementation_sha256': {file.relative_to(ROOT).as_posix(): digest(file) for file in files},
        'tongyi_port': args.tongyi_port, 'reranker_port': args.reranker_port,
    }


def run_experiment(args):
    output = args.output.resolve()
    questions, evidence = load_questions()
    needs_reranker = args.phase != 'pilot-bm25'
    server, reranker_url, reranker = preflight(args.tongyi_port, args.reranker_port, needs_reranker)
    revisions = benchmark_revisions(DATA, ('benchmark', 'indexes/bm25', 'corpus'))
    settings = experiment_settings(output, args, revisions, server, reranker, args.max_llm_calls)
    if args.phase == 'full':
        selected = select_questions(questions, evidence)
        plan = {
            'settings': settings, 'sample_seed': SAMPLE_SEED, 'question_ids': selected,
            'strata': [{'min': lo, 'max': hi, 'count': count} for lo, hi, count in STRATA],
            'excluded_pilot_ids': list(PILOT_IDS), 'methods': list(METHODS),
        }
        if (output / 'manifest.json').is_file() and read_json(output / 'manifest.json') != plan:
            raise ValueError('The saved formal sample or experimental settings differ. Use the original setup to resume.')
        write_json(output / 'manifest.json', plan)
        write_queries(output / 'queries.tsv', [questions[qid] for qid in selected])
        schedule = [
            (qid, METHODS[(number + offset) % len(METHODS)])
            for number, qid in enumerate(selected) for offset in range(len(METHODS))
        ]
        runs = output
    elif args.phase == 'pilot-bm25':
        schedule = [(qid, 'bm25') for qid in PILOT_IDS]
        runs = output / 'pilot'
    else:
        schedule = [(PILOT_IDS[0], method) for method in METHODS[1:]]
        runs = output / 'pilot'
    write_json(runs / f'{args.phase}-settings.json', settings)

    os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ['MAX_LLM_CALL_PER_RUN'] = str(args.max_llm_calls)
    configure_java()
    sys.path.insert(0, str(OFFICIAL / 'search_agent'))
    from tongyi_utils.react_agent import MultiTurnReactAgent
    from tongyi_utils.tool_search import SearchToolHandler
    from .agent_search import AgentSearch
    from .bm25 import BM25Retriever

    bm25 = BM25Retriever(DATA / 'indexes' / 'bm25')
    if bm25.document_count != 100_195:
        raise ValueError('The BM25 index must contain the complete corpus.')
    retrievers = {'bm25': bm25}
    tools = {}
    for number, (qid, method) in enumerate(schedule, 1):
        file = run_path(runs, qid, method)
        if file.is_file() and read_json(file)['termination'] not in SERVICE_ERRORS:
            continue
        base = method.removesuffix('_rerank')
        if base not in retrievers:
            from .bge import INDEX_DIR, load_retriever
            retrievers['bge'] = load_retriever(INDEX_DIR, revisions['corpus'], 'cpu')
        if method not in tools:
            adapter = AgentSearch(retrievers[base], bm25.searcher, reranker_url=reranker_url,
                                  reranker=reranker if method.endswith('_rerank') else None)
            tools[method] = SearchToolHandler(adapter, snippet_max_tokens=SNIPPET_TOKENS, k=TOP_K)
        generation = dict(GENERATION, prompt_date=settings['prompt_date'], seed=SAMPLE_SEED + int(qid))
        agent = MultiTurnReactAgent(llm={'model': MODEL, 'generate_cfg': generation},
                                    search_tool_handler=tools[method])
        print(f'[{number}/{len(schedule)}] Starting {qid} {method}', flush=True)
        item = {'question': questions[qid].text, 'answer': ''}
        result = agent._run({'item': item, 'planning_port': args.tongyi_port}, MODEL, trace_searches=True)
        result.pop('answer')
        result.update(metric_fields(result, set(evidence[qid])), query_id=qid, method=method, saved_at=utc_now())
        if file.is_file():
            archive = runs / 'failed-attempts' / qid / f'{method}-{file.stat().st_mtime_ns}.json'
            write_json(archive, read_json(file))
        write_json(file, result)
        print(f'[{number}/{len(schedule)}] Saved {qid} {method}: {result["termination"]}; '
              f'{result["valid_searches"]} searches; recall={result["evidence_recall"]:.3f}; '
              f'{result["elapsed_seconds"]:.1f}s', flush=True)
        if result['termination'] in SERVICE_ERRORS:
            raise RuntimeError(result.get('error') or result['termination'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot-bm25', 'pilot-tools', 'full', 'report'), required=True)
    parser.add_argument('--output', type=Path, default=RESULTS / 'tongyi-agent-30')
    parser.add_argument('--tongyi-port', type=int, default=16008)
    parser.add_argument('--reranker-port', type=int, default=16006)
    parser.add_argument('--max-llm-calls', type=int, default=12)
    args = parser.parse_args()
    if args.max_llm_calls < 2:
        parser.error('--max-llm-calls must be at least 2')
    if args.phase == 'report':
        from .agent_report import generate_report
        generate_report(args.output.resolve())
    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
