"""Independent artifact audit. Does not import project metric or fusion implementations."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/trust-audit-20261105'
RUNS = ['trec24-subset-20261027', 'trec24-new80-20261029',
        'trec24-100q-20261029', 'trec24-100q-rrf-20261105']


def read(file):
    return json.loads(Path(file).read_text(encoding='utf-8-sig'))


def rows(file):
    with Path(file).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def priority(seed, identifier):
    return hashlib.sha256(json.dumps([seed, identifier], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def qrels(file):
    result = {}
    for line in Path(file).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        qid, _, doc, grade = line.split()
        grade = int(grade)
        if grade < 0:
            continue
        if doc in result.setdefault(qid, {}) and result[qid][doc] != grade:
            raise ValueError('Conflicting labels')
        result[qid][doc] = grade
    return result


class Audit:
    def __init__(self):
        self.checks = Counter()
        self.failures = []
        self.max_error = 0.0

    def check(self, category, ok, detail=''):
        self.checks[category] += 1
        if not ok:
            self.failures.append({'category': category, 'detail': detail})

    def numeric(self, category, expected, actual, detail=''):
        if expected is None or actual is None:
            self.check(category, expected is actual, detail)
        else:
            error = abs(expected-actual)
            self.max_error = max(self.max_error, error)
            self.check(category, math.isfinite(actual) and error <= 1e-10, detail)


def independent_metrics(hits, labels, corpus, cfg):
    ids = [h['doc_id'] for h in hits]
    relevant = {d for d,g in labels.items() if g >= cfg['relevant_threshold']}
    available = relevant & corpus
    ideal = sorted([g for d,g in labels.items() if g > 0 and
                    (cfg.get('ndcg_scope','all_judgments') == 'all_judgments' or d in corpus)], reverse=True)
    gain = lambda g: g if cfg['gain']=='linear' else 2**g-1
    result = {'judged_query':bool(labels), 'relevant_total':len(relevant),
              'relevant_in_corpus':len(available),
              'corpus_relevant_coverage':len(available)/len(relevant) if relevant else None,
              'precision@5':len(set(ids[:5])&relevant)/5 if relevant else None}
    for cutoff in cfg['ks']:
        top = ids[:cutoff]
        dcg = math.fsum(gain(max(labels.get(d,0),0))/math.log2(i+2) for i,d in enumerate(top))
        idcg = math.fsum(gain(g)/math.log2(i+2) for i,g in enumerate(ideal[:cutoff]))
        positive_ranks = [i+1 for i,d in enumerate(top) if d in relevant]
        result.update({f'ndcg@{cutoff}':dcg/idcg if idcg else None,
                       f'mrr@{cutoff}':(1/positive_ranks[0] if positive_ranks else 0.0) if relevant else None,
                       f'recall@{cutoff}':len(set(top)&relevant)/len(relevant) if relevant else None,
                       f'judged@{cutoff}':len(set(top)&set(labels))/cutoff,
                       f'returned@{cutoff}':len(top)})
    return result


def static_audit():
    audit = Audit()
    official = dict(line.split('\t',1) for line in (ROOT/'data/raw/topics.rag24.test.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip())
    labels_all = qrels(ROOT/'data/raw/qrels.rag24.umbrela.txt')
    pool = {}
    pool_queries = set()
    candidate_records = 0
    for row in rows(ROOT/'data/raw/rag24-test-candidate-pool.jsonl'):
        query = row['query']
        pool_queries.add(str(query['qid']))
        audit.check('pool_query_text', query['text'].strip().casefold() == official[str(query['qid'])].strip().casefold(), str(query['qid']))
        for candidate in row['candidates']:
            candidate_records += 1
            doc, content = candidate['docid'], candidate['doc']
            normalized = {'doc_id':doc,'text':content['segment'].strip(),'title':str(content.get('title','')),
                          'url':str(content.get('url','')),'parent_id':str(candidate.get('parent_id',doc.split('#')[0]))}
            if doc in pool:
                audit.check('pool_duplicate_content',pool[doc]==normalized,doc)
            pool[doc] = normalized
    expected_docs = set(sorted(pool, key=lambda d:priority(42,d))[:10000])
    selected = sorted(pool_queries,key=lambda q:priority(42,q))
    selection = read(ROOT/'data/experiments/trec24-new80-20261029/selection.json')
    audit.check('topic_sampling',selection['original20']==selected[:20])
    audit.check('topic_sampling',selection['new80']==selected[20:100])
    audit.check('topic_sampling',selection['all100']==selected[:100])
    run_data, statistics_by_run = {}, {}
    for name in RUNS:
        print('Audit',name,flush=True)
        run = ROOT/'results'/name
        cfg, manifest = read(run/'config.json'),read(run/'dataset.json')
        docs_list, queries, records = rows(run/'corpus.jsonl'),rows(run/'queries.jsonl'),rows(run/'retrieval.jsonl')
        docs = {d['doc_id']:d for d in docs_list}
        topic_ids = [q['qid'] for q in queries]
        expected_topics = selected[:20] if 'subset' in name else selected[20:100] if 'new80' in name else selected[:100]
        audit.check('topic_selection',topic_ids==expected_topics,name)
        audit.check('corpus_sampling',set(docs)==expected_docs and len(docs_list)==10000,name)
        for doc, content in docs.items():
            audit.check('source_passage_text',content==pool[doc],name+':'+doc)
        for q in queries:
            audit.check('official_topic_text',q['text']==official[q['qid']],q['qid'])
        for filename,field in [('corpus.jsonl','corpus_sha256'),('queries.jsonl','queries_sha256'),('qrels.tsv','qrels_sha256')]:
            audit.check('manifest_hash',sha(run/filename)==manifest[field],name+':'+filename)
        labels = qrels(run/'qrels.tsv')
        audit.check('labels_match_official',labels=={q:labels_all[q] for q in topic_ids},name)
        audit.check('record_queries',len(records)==len(queries) and {r['qid']:r['query'] for r in records}=={q['qid']:q['text'] for q in queries},name)
        metrics = read(run/'metrics.json')
        methods = set(records[0]['methods'])
        audit.check('metrics_methods',methods==set(metrics['aggregate'])==set(metrics['per_query']),name)
        computed = {m:[] for m in methods}
        for record in records:
            qid = record['qid']
            audit.check('uniform_methods',set(record['methods'])==methods,qid)
            for method,data in record['methods'].items():
                hits = data['hits']
                ids = [h['doc_id'] for h in hits]
                audit.check('rank_integrity',len(hits)==100 and len(set(ids))==100 and set(ids)<=set(docs)
                            and [h['rank'] for h in hits]==list(range(1,101)),name+':'+qid+':'+method)
                audit.check('score_order',all(math.isfinite(h['score']) for h in hits) and
                            hits==sorted(hits,key=lambda h:(-h['score'],h['doc_id'])),name+':'+qid+':'+method)
                result = independent_metrics(hits,labels[qid],set(docs),cfg['evaluation'])
                computed[method].append(result)
                for metric,value in result.items():
                    audit.numeric('per_query_metrics',value,metrics['per_query'][method][qid][metric],name+':'+qid+':'+method+':'+metric)
                extra = data.get('fusion_ms',0)
                audit.numeric('timing_arithmetic',data['retrieval_ms']+data['rerank_ms']+extra,data['total_ms'],name+':'+qid+':'+method)
                audit.check('timing_finite',all(math.isfinite(data[k]) and data[k]>=0 for k in ('retrieval_ms','rerank_ms','total_ms')),qid)
            for base in ('bm25','bge'):
                original = {h['doc_id']:h for h in record['methods'][base]['hits']}
                ranked = record['methods'][base+'_rerank']['hits']
                audit.check('rerank_candidate_set',set(original)=={h['doc_id'] for h in ranked},qid)
                for h in ranked:
                    audit.check('rerank_provenance',h['retrieval_rank']==original[h['doc_id']]['rank']
                                and h['retrieval_score']==original[h['doc_id']]['score'] and h['rerank_score']==h['score'],qid)
            if 'rrf' in methods:
                ranks = {m:{h['doc_id']:h['rank'] for h in record['methods'][m]['hits']} for m in ('bm25','bge')}
                scores = {d:sum(1/(60+ranks[m][d]) for m in ranks if d in ranks[m]) for d in ranks['bm25'].keys()|ranks['bge'].keys()}
                expected = sorted(scores,key=lambda d:(-scores[d],d))[:100]
                fused = record['methods']['rrf']
                audit.check('rrf_full_order',[h['doc_id'] for h in fused['hits']]==expected,qid)
                audit.check('rrf_union',fused['candidate_union_count']==len(scores),qid)
                for h in fused['hits']:
                    audit.numeric('rrf_score',scores[h['doc_id']],h['score'],qid)
                    for m in ranks:
                        audit.check('rrf_source_rank',h[m+'_rank']==ranks[m].get(h['doc_id']),qid)
        for method,values in computed.items():
            for metric in values[0]:
                if metric=='judged_query':continue
                usable = [v[metric] for v in values if v[metric] is not None]
                aggregate=metrics['aggregate'][method][metric]
                audit.numeric('aggregate_metrics',statistics.mean(usable) if usable else None,aggregate['mean'],name+':'+method+':'+metric)
                audit.check('valid_query_count',aggregate['valid_queries']==len(usable),name)
            for stage in ('retrieval_ms','rerank_ms','total_ms'):
                audit.numeric('aggregate_timing',statistics.mean(r['methods'][method][stage] for r in records),metrics['aggregate'][method][stage]['mean'],name)
            trec_lines = (run/f'{method}.trec').read_text(encoding='utf-8').splitlines()
            expected_lines = [(r['qid'],h) for r in records for h in r['methods'][method]['hits']]
            audit.check('trec_count',len(trec_lines)==len(expected_lines),name+method)
            for line,(qid,h) in zip(trec_lines,expected_lines):
                tq,_,doc,rank,score,tag=line.split()
                audit.check('trec_export',tq==qid and doc==h['doc_id'] and int(rank)==h['rank'] and tag==method
                            and abs(float(score)-1/h['rank'])<1e-11,name+qid+method)
        parsed_pairs=read(run/'paired_analysis.json')['comparisons']
        for pair in parsed_pairs.values():
            deltas=[metrics['per_query'][pair['other']][qid]['ndcg@10']-metrics['per_query'][pair['base']][qid]['ndcg@10'] for qid in topic_ids]
            audit.check('pair_counts',pair['wins']==sum(x>1e-12 for x in deltas) and pair['losses']==sum(x< -1e-12 for x in deltas)
                        and pair['ties']==sum(abs(x)<=1e-12 for x in deltas),name)
        stats={m:{k:metrics['aggregate'][m][k]['mean'] for k in ('ndcg@5','ndcg@10','recall@100','judged@10','total_ms')} for m in methods}
        statistics_by_run[name]={'queries':len(queries),'documents':len(docs),'ndcg_scope':cfg['evaluation'].get('ndcg_scope','all_judgments'),
                                 'mean_relevant_coverage':statistics.mean(len({d for d,g in labels[q].items() if g>=1}&set(docs))/len({d for d,g in labels[q].items() if g>=1}) for q in topic_ids),
                                 'statistics':stats}
        run_data[name]={'records':records,'metrics':metrics,'docs':docs,'queries':queries}
    original=run_data[RUNS[2]]
    sources={r['qid']:r for name in RUNS[:2] for r in run_data[name]['records']}
    audit.check('batch_merge',original['records']==[sources[q['qid']] for q in original['queries']])
    for record in run_data[RUNS[3]]['records']:
        src=sources[record['qid']]
        for method in src['methods']:
            audit.check('new_old_hit_equality',record['methods'][method]['hits']==src['methods'][method]['hits'],record['qid'])
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'passed':not audit.failures,
            'checks':dict(audit.checks),'total_checks':sum(audit.checks.values()),'max_numeric_absolute_error':audit.max_error,
            'failures':audit.failures,'raw_data':{'official_topics':len(official),'pool_queries':len(pool_queries),
            'candidate_records':candidate_records,'unique_candidates':len(pool),'grade_pair_counts':dict(Counter(g for q in labels_all.values() for g in q.values()))},
            'runs':statistics_by_run}
    OUT.mkdir(exist_ok=True)
    (OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    static_audit()
