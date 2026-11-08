"""Cohort summaries for composites; retain separate resource measurement periods."""
import html
from .evaluation import evaluate, ndcg_description
from .schemas import methods_for, LABELS
from .utils import read_json, write_json


def cohort_panel(run, records, labels, corpus_ids, evaluation, judged_report=False):
    METHODS=methods_for(records)
    metadata=read_json(run/'cohorts.json')
    esc=lambda value:html.escape(str(value),quote=True)
    fmt=lambda value:'N/A' if value is None else f'{value:.4f}'
    keys=['ndcg@5','ndcg@10','precision@5','mrr@10','recall@100','total_ms']
    if judged_report:
        keys=['ndcg@5','ndcg@10','mrr@10','judged@10','total_ms']
    result={}
    body='<div class="panel"><h2>原20题 / 新增80题 / 全部100题</h2><p>'+esc(metadata['note'])+'</p>'
    body+='<p class="metricnote">'+esc(ndcg_description(evaluation))+'</p>'
    for cohort in metadata['cohorts']+[{'name':'全部100题','query_ids':[r['qid'] for r in records]}]:
        wanted=set(cohort['query_ids'])
        rows=[r for r in records if r['qid'] in wanted]
        if len(rows)!=len(wanted):raise ValueError('批次问题清单与排序不一致')
        metrics=evaluate(rows,labels,corpus_ids,evaluation)
        result[cohort['name']]=metrics
        body+='<h3>'+esc(cohort['name'])+f' · {len(rows)}题</h3>'
        body+='<table><thead><tr><th>方案</th>'+''.join('<th>'+esc(k.replace('judged','Judged'))+'</th>' for k in keys)+'<th>有效题数 nDCG@10</th></tr></thead><tbody>'
        for method in METHODS:
            stats=metrics['aggregate'][method]
            body+='<tr><td>'+esc(LABELS[method])+'</td>'+''.join('<td>'+fmt(stats[k]['mean'])+'</td>' for k in keys)+f'<td>{stats["ndcg@10"]["valid_queries"]}/{len(rows)}</td></tr>'
        body+='</tbody></table>'
        if 'answer_count' in cohort:
            body+=f'<p class="metricnote">已生成 {cohort["answer_count"]}/{len(rows)*4} 份回答；'
            body+='<a href="../../'+esc(cohort['run_id'])+'/report/index.html">查看该批次报告</a>。total_ms单位为ms。</p>'
        performance=cohort.get('performance')
        if performance:
            body+='<p class="metricnote">该批次生成资源实测：整卡峰值 '+fmt(performance.get('peak_gpu_used_mib'))+' MiB；最低可用内存 '+fmt(performance.get('min_system_ram_available_mib'))+' MiB。两批次不是连续测量，不合并为一次全程资源实测。</p>'
    write_json(run/'cohort_metrics.json',result)
    return body+'</div>'
