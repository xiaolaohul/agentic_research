"""Expand a frozen 20-topic experiment with 80 new topics; retain source fingerprints."""
from pathlib import Path
import copy
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMP = ROOT / '.cache/tmp'
TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(TEMP)
os.environ['HF_HOME'] = str(ROOT / 'models/huggingface')
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.utils import (read_json, write_json, write_jsonl, jsonl, digest, object_hash,
                       utc_now, load_config, path, versions, atomic_text)
from src.dataset import load_queries, load_documents, load_qrels, save_prepared
from src.schemas import METHODS

OLD = ROOT / 'results/trec24-subset-20261027'
NEW_ID = 'trec24-new80-20261029'
TOTAL_ID = 'trec24-100q-20261029'
NEW = ROOT / 'results' / NEW_ID
TOTAL = ROOT / 'results' / TOTAL_ID
CONFIG = ROOT / 'configs/trec24-new80-20261029.yaml'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_source(run):
    state = read_json(run / 'run.json')
    require(state['status'] == 'complete', f'{run.name}: retrieval incomplete')
    manifest = read_json(run / 'dataset.json')
    for file, key in [('corpus.jsonl', 'corpus_sha256'), ('queries.jsonl', 'queries_sha256'), ('qrels.tsv', 'qrels_sha256')]:
        require(digest(run / file) == manifest[key], f'{run.name}: {file} snapshot mismatch')
    return state, manifest, read_json(run / 'config.json')


def prepare():
    import yaml
    _, manifest, cfg = checked_source(OLD)
    require(versions() == read_json(OLD/'run.json')['versions'], 'Package versions differ from original experiment')
    oldq = load_queries(OLD/'queries.jsonl')
    official = load_queries(ROOT/'data/raw/topics.rag24.test.txt')
    available = {str(r['query']['qid']) for r in jsonl(ROOT/'data/raw/rag24-test-candidate-pool.jsonl')}
    oldids = {q.qid for q in oldq}
    require(len(oldids) == 20, 'Expected original 20 topics')
    official_map = {q.qid:q.text for q in official}
    require(all(official_map.get(q.qid) == q.text for q in oldq), 'Original question differs from official text')
    # Labels are not read until after deterministic selection.
    eligible = sorted((q for q in official if q.qid not in oldids and q.qid in available),
                      key=lambda q: object_hash([cfg['dataset']['seed'], q.qid]))
    newq = eligible[:80]
    require(len(newq) == 80 and not oldids.intersection(q.qid for q in newq), 'Need 80 unique new topics')
    selection = {'created_at':utc_now(), 'source_run':OLD.name,
                 'sampling':'SHA256 priority [seed,qid], excluding original20; no qrels used',
                 'seed':cfg['dataset']['seed'], 'original20':[q.qid for q in oldq],
                 'new80':[q.qid for q in newq], 'all100':[q.qid for q in oldq+newq],
                 'corpus_sha256':manifest['corpus_sha256'],
                 'source_preserved':{n:digest(OLD/n) for n in ['retrieval.jsonl','answers.jsonl','performance.json','config.json','dataset.json'] if (OLD/n).is_file()}}
    folder = ROOT/'data/experiments'/NEW_ID
    folder.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg.pop('generation', None)
    for key, file in [('corpus','corpus.jsonl'),('queries','queries.jsonl'),('qrels','qrels.tsv'),('manifest','manifest.json')]:
        cfg['dataset'][key] = str((folder/file).relative_to(ROOT)).replace('\\','/')
    cfg['dataset']['name'] = 'trec-rag-2024-fixed-corpus-new80'
    cfg['dataset']['max_queries'] = 80
    if (folder/'manifest.json').exists():
        require(read_json(folder/'selection.json')['new80'] == selection['new80'], 'Selection changed')
    else:
        provenance = copy.deepcopy(manifest['provenance'])
        provenance['expansion'] = {'original_run':OLD.name,'original20':selection['original20'],'new80':selection['new80']}
        prepared = save_prepared(cfg,load_documents(OLD/'corpus.jsonl'),newq,
                                 load_qrels(ROOT/'data/raw/qrels.rag24.umbrela.txt'),provenance)
        require(prepared['corpus_sha256'] == manifest['corpus_sha256'], 'Corpus differs')
        write_json(folder/'selection.json',selection)
    if CONFIG.exists():
        require(load_config(CONFIG) == cfg, 'New80 config changed')
    else:
        atomic_text(CONFIG,yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    print('Prepared fixed corpus, original20 + disjoint new80; TEMP/TMP on E:',TEMP,flush=True)


def merge(verify=False):
    import yaml
    from tools.audit_100 import historical_messages
    from src.report import generate_report
    from src.pipeline import export_metrics_csv
    old_state, old_manifest, old_cfg = checked_source(OLD)
    new_state, new_manifest, new_cfg = checked_source(NEW)
    retrieval_only = new_state.get('report_mode') == 'retrieval_only'
    require(old_manifest['corpus_sha256'] == new_manifest['corpus_sha256'], 'Sources use different corpus')
    for key in ('retrieval','reranker','evaluation'):
        require(old_cfg[key] == new_cfg[key], f'{key} parameters differ')
    for key in ('bge_revision','reranker_revision'):
        require(old_state[key] == new_state[key], f'{key} differs')
    oldrows, newrows = list(jsonl(OLD/'retrieval.jsonl')), list(jsonl(NEW/'retrieval.jsonl'))
    require(len(oldrows)==20 and len(newrows)==80, 'Unexpected topic count')
    require(len({r['qid'] for r in oldrows+newrows})==100,'Duplicate topics')
    lookup = {r['qid']:r for r in oldrows+newrows}
    docs = {d.doc_id:d for d in load_documents(OLD/'corpus.jsonl')}
    TOTAL.mkdir(parents=True,exist_ok=True)
    if not (TOTAL/'dataset.json').exists():
        cfg = copy.deepcopy(new_cfg)
        for key,file in [('corpus','corpus.jsonl'),('queries','queries.jsonl'),('qrels','qrels.tsv'),('manifest','dataset.json')]:
            cfg['dataset'][key] = str((TOTAL/file).relative_to(ROOT)).replace('\\','/')
        cfg['dataset']['max_queries']=100
        cfg['dataset']['name']='trec-rag-2024-fixed-corpus-100q'
        provenance=copy.deepcopy(old_manifest['provenance'])
        provenance['batches']=[OLD.name,NEW.name]
        save_prepared(cfg,list(docs.values()),load_queries(OLD/'queries.jsonl')+load_queries(NEW/'queries.jsonl'),
                      {**load_qrels(OLD/'qrels.tsv'),**load_qrels(NEW/'qrels.tsv')},provenance)
        write_json(TOTAL/'config.json',cfg)
        atomic_text(ROOT/'configs'/f'{TOTAL_ID}.yaml',yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    records = oldrows+newrows
    for r in records:
        for base in ('bm25','bge'):
            require({h['doc_id'] for h in r['methods'][base]['hits']} == {h['doc_id'] for h in r['methods'][base+'_rerank']['hits']},'Candidate set changed')
    write_jsonl(TOTAL/'retrieval.jsonl',records)
    for base in ('bm25','bge'):
        write_jsonl(TOTAL/f'{base}.candidates.jsonl',({'qid':r['qid'],**r['methods'][base]} for r in records))
    for method in METHODS:
        atomic_text(TOTAL/f'{method}.trec',''.join(f"{r['qid']} Q0 {h['doc_id']} {h['rank']} {1/h['rank']:.12f} {method}\n" for r in records for h in r['methods'][method]['hits']))
    answers=[];cohorts=[];seen=set();source_generation={}
    for run,rows,label in [(OLD,oldrows,'原20题'),(NEW,newrows,'新增80题')]:
        qids={r['qid'] for r in rows}
        gen=read_json(run/'generation.json') if (run/'generation.json').exists() else None
        source_generation[run.name]=gen
        if gen:
            require(gen['configuration']==old_cfg['generation'],'Actual generation configuration differs')
            require(gen['context_k']==old_cfg['retrieval']['context_k'],'Evidence count differs')
            if source_generation.get(OLD.name):
                require(gen['model_artifact']==source_generation[OLD.name]['model_artifact'],'Model artifact differs')
        source_answers=list(jsonl(run/'answers.jsonl')) if (run/'answers.jsonl').exists() else []
        for answer in source_answers:
            key=(answer['qid'],answer['method'])
            require(key not in seen and answer['qid'] in qids and answer['method'] in METHODS,'Unexpected/duplicate answer')
            require(gen and answer['fingerprint']==gen['fingerprint'],'Source answer fingerprint differs')
            r=lookup[answer['qid']]
            expected=[h['doc_id'] for h in r['methods'][answer['method']]['hits'][:old_cfg['retrieval']['context_k']]]
            require([e['doc_id'] for e in answer['evidence']]==expected,'Evidence IDs differ')
            require(all(docs[e['doc_id']].search_text.startswith(e['text']) for e in answer['evidence']),'Evidence text differs')
            require(answer['messages']==historical_messages(r['query'],answer['evidence']),'Prompt differs')
            seen.add(key)
            answers.append(dict(answer,source_run=run.name,source_batch=label))
        cohort={'name':label,'run_id':run.name,'query_ids':[r['qid'] for r in rows]}
        if not retrieval_only:
            cohort.update(answer_count=len(source_answers),performance=read_json(run/'performance.json') if (run/'performance.json').exists() else None)
        cohorts.append(cohort)
    if answers:
        write_jsonl(TOTAL/'answers.jsonl',answers)
    state={'run_id':TOTAL_ID,'created_at':read_json(TOTAL/'run.json')['created_at'] if (TOTAL/'run.json').exists() else utc_now(),
           'status':'complete','generation_status':'complete' if len(answers)==400 else new_state.get('generation_status', 'not_requested'),
           'bge_revision':new_state['bge_revision'],'reranker_revision':new_state['reranker_revision'],
           'versions':versions(),'composite_sources':[OLD.name,NEW.name],
           'resume_run':NEW.name,'updated_at':utc_now()}
    if retrieval_only:
        state.update(report_mode='retrieval_only',generation_status='not_requested')
    write_json(TOTAL/'run.json',state)
    note=('仅检索评测；同一10,000段落与检索参数。原20题复用原排序，新80题本次运行；耗时分别展示各批次，并重新计算100题总均值。' if retrieval_only else '同一语料与模型参数；原20题复用，新80题本次执行。答案保留各来源批次指纹。耗时为混合批次均值，资源测量按批次分别展示。')
    write_json(TOTAL/'cohorts.json',{'cohorts':cohorts,'note':note})
    if any(source_generation.values()):
        write_json(TOTAL/'generation.json',{'configuration':old_cfg['generation'],'context_k':old_cfg['retrieval']['context_k'],
                   'fingerprint_scheme':'per-source-run; retained without rewriting','source_generations':source_generation})
    generate_report(TOTAL,verify_metrics=verify)
    return len(answers)


def run():
    prepare()
    if not (NEW/'run.json').exists():
        subprocess.run([sys.executable,'-u',str(ROOT/'main.py'),'--config',str(CONFIG),'run','--run-id',NEW_ID],cwd=ROOT,check=True)
    checked_source(NEW)
    state=read_json(NEW/'run.json')
    state.update(report_mode='retrieval_only',generation_status='not_requested')
    write_json(NEW/'run.json',state)
    from src.report import generate_report
    generate_report(NEW,verify_metrics=True)
    merge(verify=True)
    selection=read_json(ROOT/'data/experiments'/NEW_ID/'selection.json')
    require(all(digest(OLD/n)==checksum for n,checksum in selection['source_preserved'].items()),'Original experiment was modified')
    write_json(TOTAL/'completion.json',{'completed_at':utc_now(),'topics':100,'retrieval_records':400,'mode':'retrieval_only','original20_unchanged':True})
    print('RETRIEVAL COMPLETE:',TOTAL/'report/index.html',flush=True)


if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'run'
    if mode=='prepare':prepare()
    elif mode=='merge':merge(verify=True)
    elif mode=='run':run()
    else:raise ValueError('Use prepare, merge or run')
