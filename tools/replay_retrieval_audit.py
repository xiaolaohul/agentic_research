"""Read-only replay of all 100 queries; write audit evidence outside original runs."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME'] = str(ROOT/'models/huggingface')
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from src.utils import read_json, jsonl, write_json, write_jsonl, release_gpu, object_hash, digest
from src.dataset import load_documents
from src.schemas import Hit
from src.pipeline import index_fingerprint
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.bge import BGERetriever
from src.reranker import Reranker

RUN = ROOT/'results/trec24-100q-rrf-20261105'
OUT = ROOT/'results/trust-audit-20261105'


def main():
    started=time.perf_counter()
    cfg=read_json(RUN/'config.json')
    manifest=read_json(RUN/'dataset.json')
    records=list(jsonl(RUN/'retrieval.jsonl'))
    docs=load_documents(RUN/'corpus.jsonl')
    original_hashes={name:digest(RUN/name) for name in ('retrieval.jsonl','metrics.json','config.json','corpus.jsonl','queries.jsonl','qrels.tsv')}
    result={'started_at':datetime.now(timezone.utc).isoformat(),'query_count':len(records),
            'offline':True,'models':{},'mismatches':[],'method_checks':{},'source_hashes':original_hashes}
    fresh_rows={r['qid']:{'qid':r['qid'],'query':r['query'],'methods':{}} for r in records}
    def compare(method, row, fresh):
        expected=row['methods'][method]['hits']
        identical_ids=[h.doc_id for h in fresh]==[h['doc_id'] for h in expected]
        saved={h['doc_id']:h for h in expected}
        common=[(h,saved[h.doc_id]) for h in fresh if h.doc_id in saved]
        error=max((abs(h.score-old['score']) for h,old in common),default=0.0)
        stats=result['method_checks'].setdefault(method,{'queries':0,'identical_rankings':0,'identical_scores':0,'max_score_error':0.0})
        stats['queries']+=1
        stats['identical_rankings']+=identical_ids
        stats['identical_scores']+=error==0.0 and len(common)==len(expected)
        stats['max_score_error']=max(stats['max_score_error'],error)
        if not identical_ids:
            result['mismatches'].append({'qid':row['qid'],'method':method,'max_common_score_error':error,
                'rank_changes':[{'rank':i+1,'saved':old['doc_id'],'replayed':new.doc_id}
                                for i,(new,old) in enumerate(zip(fresh,expected)) if new.doc_id!=old['doc_id']]})
        fresh_rows[row['qid']]['methods'][method]={'hits':[h.to_dict() for h in fresh]}
    for method,cls in [('bm25',BM25Retriever),('bge',BGERetriever)]:
        print('Replay',method,'100 queries',flush=True)
        retriever=cls(docs,cfg['retrieval'][method])
        retriever.load(ROOT/cfg['retrieval']['index_dir']/method,index_fingerprint(cfg,manifest,method))
        if method=='bge':
            retriever.load_model()
            result['models']['bge_revision']=read_json(ROOT/cfg['retrieval']['index_dir']/method/'metadata.json')['resolved_revision']
            import numpy as np
            sample=sorted(range(len(docs)),key=lambda i:object_hash([42,docs[i].doc_id]))[:16]
            vectors=retriever.model.encode([docs[i].search_text for i in sample],batch_size=16,
                                          normalize_embeddings=True,show_progress_bar=False)
            stored=np.stack([retriever.index.reconstruct(int(i)) for i in sample])
            result['index_vector_check']={'documents':16,'max_absolute_error':float(np.max(np.abs(vectors-stored))),
                                          'all_close_at_1e-5':bool(np.allclose(vectors,stored,atol=1e-5,rtol=1e-5)),
                                          'faiss_documents':retriever.index.ntotal}
        for index,row in enumerate(records,1):
            compare(method,row,retriever.retrieve(row['query'],100))
            if index%20==0:print(method,index,'/100',flush=True)
        del retriever
        release_gpu()
        write_json(OUT/'replay_progress.json',result)
    print('Replay reranker: 20,000 query-passage pairs',flush=True)
    ranker=Reranker(docs,cfg['reranker'])
    result['models']['reranker_revision']=ranker.resolved_revision
    for index,row in enumerate(records,1):
        for base in ('bm25','bge'):
            fresh=ranker.rerank(row['query'],[Hit(**h) for h in row['methods'][base]['hits']])
            compare(base+'_rerank',row,fresh)
        if index%10==0:
            print('rerank',index,'/100',flush=True)
            write_json(OUT/'replay_progress.json',result)
    del ranker
    release_gpu()
    result['source_files_unchanged']=all(digest(RUN/name)==checksum for name,checksum in original_hashes.items())
    result['identical_all_rankings']=not result['mismatches']
    result['identical_all_scores']=all(v['identical_scores']==100 for v in result['method_checks'].values())
    result['completed_at']=datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds']=time.perf_counter()-started
    write_jsonl(OUT/'replayed_rankings.jsonl',fresh_rows.values())
    write_json(OUT/'model_replay.json',result)
    print({k:v for k,v in result.items() if k not in ('mismatches','source_hashes')},flush=True)


if __name__=='__main__':
    main()
