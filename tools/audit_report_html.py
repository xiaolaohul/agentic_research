"""Check rendered five-method HTML against frozen JSONL, including full source text."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/trec24-100q-rrf-20261105'
OUT=ROOT/'results/trust-audit-20261105'
METHODS=['bm25','bge','rrf','bm25_rerank','bge_rerank']


def rows(file):
    with Path(file).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Reader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.columns=[]
        self.active=self.article=self.capture=self.qrow=self.cell=None
        self.buffer=[]
        self.bars=[]
        self.queryrows=[]

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        classes=a.get('class','').split()
        if tag=='section' and 'column' in classes:
            self.active=[]
            self.columns.append(self.active)
        if tag=='article' and 'hit' in classes:
            self.article={'doc_id':a['data-doc']}
            self.active.append(self.article)
        if self.article is not None:
            if tag=='span' and 'badge' in classes and set(classes)&{'relevant','nonrelevant','unjudged'}:
                self.capture='grade'
                self.buffer=[]
            if tag=='div' and 'fulltext' in classes:
                self.capture='text'
                self.buffer=[]
        if 'barrow' in classes and 'data-metrics' in a:
            self.bars.append(json.loads(a['data-metrics']))
        if tag=='tr' and 'data-query' in a:
            self.qrow=[]
            self.queryrows.append(self.qrow)
        if tag=='td' and self.qrow is not None:
            self.cell=[]

    def handle_data(self,text):
        if self.capture:
            self.buffer.append(text)
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self,tag):
        if (tag=='span' and self.capture=='grade') or (tag=='div' and self.capture=='text'):
            self.article[self.capture]=''.join(self.buffer)
            self.capture=None
        if tag=='article':
            self.article=None
        if tag=='section':
            self.active=None
        if tag=='td' and self.cell is not None:
            self.qrow.append(''.join(self.cell))
            self.cell=None
        if tag=='tr':
            self.qrow=None


def main():
    metrics=json.loads((RUN/'metrics.json').read_text(encoding='utf-8'))
    docs={r['doc_id']:r for r in rows(RUN/'corpus.jsonl')}
    records=rows(RUN/'retrieval.jsonl')
    labels={}
    for line in (RUN/'qrels.tsv').read_text(encoding='utf-8').splitlines():
        q,_,d,g=line.split()
        labels.setdefault(q,{})[d]=int(g)
    reader=Reader()
    reader.feed((RUN/'report/index.html').read_text(encoding='utf-8'))
    assert len(reader.bars)==5 and len(reader.queryrows)==100
    for method,bar in zip(METHODS,reader.bars):
        for metric,value in bar.items():
            assert value==metrics['aggregate'][method][metric]['mean']
    for record,cells in zip(records,reader.queryrows):
        assert record['qid'] in cells[0]
        for method,value in zip(METHODS,cells[1:6]):
            assert value==f'{metrics["per_query"][method][record["qid"]]["ndcg@10"]:.4f}'
    count=0
    for record in records:
        qid=record['qid']
        name='q-'+hashlib.sha256(json.dumps(qid,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]+'.html'
        page=Reader()
        page.feed((RUN/'report'/name).read_text(encoding='utf-8'))
        assert len(page.columns)==5
        for method,column in zip(METHODS,page.columns):
            assert len(column)==100
            for saved,shown in zip(record['methods'][method]['hits'],column):
                doc=saved['doc_id']
                grade=labels[qid].get(doc)
                assert shown['doc_id']==doc
                assert shown['text']==docs[doc]['text'].replace('\r\n','\n').replace('\r','\n')
                assert shown['grade']==('未标注' if grade is None else f'相关性 {grade}' if grade>=1 else f'已标注 {grade}')
                count+=1
    report={'passed':True,'overview_method_bars':5,'overview_query_rows':100,'detail_pages':100,
            'displayed_passages_text_grade_order_checked':count}
    (OUT/'html_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':
    main()
