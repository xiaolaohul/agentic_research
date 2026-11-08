"""Supplement frozen four-method runs with an auditable, CPU-only RRF baseline."""
import copy
import re
import shutil
from pathlib import Path

from .dataset import validate_prepared, load_documents, load_queries
from .evaluation import evaluate
from .retrieval.rrf import fused_result
from .schemas import METHODS, HYBRID_METHODS, methods_for
from .utils import digest, jsonl, path, read_json, utc_now, write_json, write_jsonl, atomic_text, storage_check, versions


def supplement_rrf(source, cfg, run_id):
    source = Path(source)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('实验编号只能包含字母、数字、下划线和连字符。')
    source_state = read_json(source / 'run.json')
    if source_state['status'] != 'complete':
        raise ValueError('来源实验未完成。')
    manifest, source_cfg = read_json(source / 'dataset.json'), read_json(source / 'config.json')
    for name, key in [('corpus.jsonl', 'corpus_sha256'), ('queries.jsonl', 'queries_sha256'), ('qrels.tsv', 'qrels_sha256')]:
        if digest(source / name) != manifest[key]:
            raise ValueError(f'来源实验快照已变化：{name}')
    if any(cfg[key] != source_cfg[key] for key in ('evaluation',)):
        raise ValueError('评测口径与来源实验不一致，拒绝混合。')
    if {k:v for k,v in cfg['retrieval'].items() if k != 'rrf'} != source_cfg['retrieval']:
        raise ValueError('基础检索配置与来源实验不一致。')
    if {k:v for k,v in cfg['reranker'].items() if k != 'depth'} != {k:v for k,v in source_cfg['reranker'].items() if k != 'depth'}:
        raise ValueError('重排模型配置与来源实验不一致。')
    fusion = cfg['retrieval'].get('rrf', {})
    if not fusion.get('enabled'):
        raise ValueError('请使用启用RRF的配置。')
    depth = cfg['reranker'].get('depth', cfg['retrieval']['candidate_k'])
    if depth != source_cfg['retrieval']['candidate_k']:
        raise ValueError('复用的历史结果必须是全部候选重排，深度需要与来源候选数一致。')
    rows = list(jsonl(source / 'retrieval.jsonl'))
    if set(methods_for(rows)) != set(METHODS):
        raise ValueError('补充入口只接受四组基础实验。')
    queries = load_queries(source / 'queries.jsonl')
    if len(rows) != len(queries) or {r['qid']:r['query'] for r in rows} != {q.qid:q.text for q in queries}:
        raise ValueError('来源排序的问题清单与问题快照不一致。')
    docs = {d.doc_id for d in load_documents(source / 'corpus.jsonl')}
    for row in rows:
        for method, result in row['methods'].items():
            hits = result['hits']
            if (len(hits) != depth or len({h['doc_id'] for h in hits}) != depth
                    or any(h['doc_id'] not in docs for h in hits)
                    or [h['rank'] for h in hits] != list(range(1, depth+1))):
                raise ValueError('来源候选数量、名次或段落ID不符合记录。')
        for base in ('bm25', 'bge'):
            original = {h['doc_id']:h for h in row['methods'][base]['hits']}
            reranked = row['methods'][base+'_rerank']['hits']
            if set(original) != {h['doc_id'] for h in reranked} or any(
                h['rerank_score'] is None or h['retrieval_rank'] != original[h['doc_id']]['rank']
                or h['retrieval_score'] != original[h['doc_id']]['score'] for h in reranked):
                raise ValueError('来源重排未完整保留原候选信息。')
    source_hashes = {str(f.relative_to(source)):digest(f) for f in source.rglob('*') if f.is_file()}
    run = path(cfg['results_dir']) / run_id
    storage_check(cfg, sum((source/f).stat().st_size for f in ('corpus.jsonl','queries.jsonl','qrels.tsv','retrieval.jsonl')))
    run.mkdir(parents=True, exist_ok=False)
    state = {'run_id':run_id, 'created_at':utc_now(), 'status':'running', 'generation_status':'not_requested',
             'report_mode':'retrieval_only', 'report_profile':source_state.get('report_profile'),
             'show_cohort_summary':False, 'show_pair_comparison':True,
             'methods':list(HYBRID_METHODS), 'source_run':source.name, 'source_path':str(source.resolve()),
             'versions':versions(), 'source_versions':source_state['versions'], 'rerank_depth':depth,
             'bge_revision':source_state.get('bge_revision'), 'reranker_revision':source_state.get('reranker_revision'),
             'timing_note':'原四组保留来源批次测量；RRF总耗时为历史两路检索耗时之和加本次实测融合耗时，属于串行估算。'}
    write_json(run/'run.json', state)
    try:
        new_cfg = copy.deepcopy(cfg)
        for key, filename in [('corpus','corpus.jsonl'),('queries','queries.jsonl'),('qrels','qrels.tsv'),('manifest','dataset.json')]:
            new_cfg['dataset'][key] = str((run/filename).resolve())
        write_json(run/'config.json', new_cfg)
        write_json(run/'dataset.json', manifest)
        for filename in ('corpus.jsonl','queries.jsonl','qrels.tsv'):
            shutil.copyfile(source/filename, run/filename)
        for row in rows:
            for base in ('bm25','bge'):
                row['methods'][base+'_rerank'].update(rerank_depth=depth, reranked_count=depth,
                    depth_provenance='source candidate count, set equality and per-hit rerank scores verified')
            row['methods']['rrf'] = fused_result(row['methods'], fusion, reused=True)
        write_jsonl(run/'retrieval.jsonl', rows)
        for method in ('bm25','bge','rrf'):
            write_jsonl(run/f'{method}.candidates.jsonl', ({'qid':r['qid'], **r['methods'][method]} for r in rows))
        for method in HYBRID_METHODS:
            atomic_text(run/f'{method}.trec', ''.join(
                f"{r['qid']} Q0 {h['doc_id']} {h['rank']} {1/h['rank']:.12f} {method}\n"
                for r in rows for h in r['methods'][method]['hits']))
        _, _, labels, _ = validate_prepared(new_cfg)
        metrics = evaluate(rows, labels, docs, new_cfg['evaluation'])
        old_metrics = read_json(source/'metrics.json')
        if any(metrics[scope][method] != old_metrics[scope][method]
               for scope in ('aggregate','per_query') for method in METHODS):
            raise ValueError('原四组指标与来源实验不一致。')
        originals = {r['qid']:r for r in jsonl(source/'retrieval.jsonl')}
        if any(row['methods'][method]['hits'] != originals[row['qid']]['methods'][method]['hits']
               for row in rows for method in METHODS):
            raise ValueError('原四组排名或分数发生变化。')
        write_json(run/'metrics.json', metrics)
        # Reuse human explanations only after checking their complete numerical cases.
        if source_state.get('show_rerank_drop_analysis'):
            from .rerank_drops import render_drop_panel, extract_cases, provenance
            render_drop_panel(source)
            if extract_cases(source) != extract_cases(run):
                raise ValueError('来源退步分析案例已变化。')
            drops = read_json(source/'rerank_drop10.json')
            drops['reused_source'] = {'run':source.name, 'sha256':digest(source/'rerank_drop10.json'),
                                      'case_equality_verified':True}
            drops['provenance'] = provenance(run)
            write_json(run/'rerank_drop10.json', drops)
            state['show_rerank_drop_analysis'] = True
            write_json(run/'run.json', state)
        from .report import generate_report
        generate_report(run, verify_metrics=True)
        unchanged = source_hashes == {str(f.relative_to(source)):digest(f) for f in source.rglob('*') if f.is_file()}
        if not unchanged:
            raise ValueError('来源实验在操作期间发生变化。')
        write_json(run/'artifact_audit.json', {'checked_at':utc_now(), 'source_run':source.name,
            'source_files_sha256':source_hashes, 'original_source_unchanged':unchanged,
            'original_four_metrics_identical':True, 'original_four_hit_lists_identical':True,
            'queries':len(rows), 'documents':len(docs), 'method_query_records':len(rows)*len(HYBRID_METHODS),
            'rrf':fusion, 'rerank_depth':depth, 'model_inference_performed':False})
        state.update(status='complete', completed_at=utc_now())
        write_json(run/'completion.json', {'completed_at':state['completed_at'], 'topics':len(rows),
                                          'retrieval_records':len(rows)*len(HYBRID_METHODS), 'mode':'retrieval_only'})
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(run/'run.json', state)
    return run
