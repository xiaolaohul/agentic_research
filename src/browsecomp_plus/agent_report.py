"""Write the offline HTML overview and per-question Agent comparison pages."""

from collections import Counter
from html import escape
import json
from statistics import mean, median

from .evaluation import QRELS_FILES
from .run_agent import OFFICIAL, SERVICE_ERRORS, run_path
from ..dataset import load_qrels
from ..utils import atomic_text, read_json

LABELS = {'bm25': 'BM25', 'bm25_rerank': 'BM25 + 重排', 'bge': 'BGE', 'bge_rerank': 'BGE + 重排'}
COLORS = {'bm25': '#356fd0', 'bm25_rerank': '#155d51', 'bge': '#8853b0', 'bge_rerank': '#b34c26'}
METRICS = (
    ('证据召回', 'evidence_recall', 'percent'), ('有效搜索', 'valid_searches', 'count'),
    ('输入 token', 'prompt_tokens', 'count'), ('输出 token', 'completion_tokens', 'count'),
    ('总 token', 'total_tokens', 'count'), ('整题耗时', 'elapsed_seconds', 'seconds'),
)
STYLE = """
*{box-sizing:border-box} body{margin:0;background:#f3f6fa;color:#18253c;
font:15px/1.55 system-ui,sans-serif} main{max-width:1340px;margin:auto;padding:32px 24px 80px}
h1,h2,h3,p{margin:.3em 0 .65em} h1{font-size:clamp(25px,4vw,38px)}
h2{margin-top:1.35em} a{color:#2155a5} .muted{color:#475976} .grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:15px}
article,details,.box{border:1px solid #d8e0ee;border-radius:12px;background:white;padding:14px}
article{min-width:0} details{margin:10px 0} summary{cursor:pointer;font-weight:650;overflow-wrap:anywhere}
.scroll{overflow:auto} table{border-collapse:collapse;width:100%;text-align:left}
th,td{padding:9px 10px;vertical-align:top;border-bottom:1px solid #e2e8f0}
th{background:#edf2f8} .barrow{display:grid;grid-template-columns:105px 1fr auto;
align-items:center;gap:9px;margin:8px 0} .track{height:11px;border-radius:6px;background:#e8edf6}
.bar{display:block;height:100%;border-radius:6px} pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;
background:#f4f7fb;padding:10px;border-radius:8px} .pill{font-size:13px;color:#425577}
.rounds{list-style:none;margin:0;padding:0}.rounds li{border-top:1px solid #e2e8f0;padding:8px 0}
.round-query{display:block;overflow-wrap:anywhere}
.round-counts{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:3px}
.delta-up{color:#145e4d}.delta-down{color:#9c352a}
input[type=search]{font:inherit;width:min(100%,360px);padding:7px 10px;margin-bottom:12px}
"""


def esc(value):
    return escape(str(value), quote=True)


def show(value, kind):
    if value is None:
        return '—'
    if kind == 'percent':
        return f'{value:.1%}'
    if kind == 'seconds':
        return f'{value:.1f} 秒'
    return f'{value:,.0f}' if value == int(value) else f'{value:,.1f}'


def metric(record, key):
    if key.endswith('_tokens'):
        return record['token_usage'][key]
    return record.get(key)


def html_page(title, body):
    return (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(title)}</title><style>{STYLE}</style><main>{body}</main></html>\n')


def chart(title, by_method, kind):
    scale = 1 if kind == 'percent' else max((value or 0 for value in by_method.values()), default=0)
    rows = []
    for method, value in by_method.items():
        width = value / scale * 100 if value is not None and scale else 0
        rows.append(f'<div class="barrow"><span>{LABELS[method]}</span><span class="track" aria-hidden="true">'
                    f'<span class="bar" style="width:{width:.3f}%;background:{COLORS[method]}"></span>'
                    f'</span><strong>{show(value, kind)}</strong></div>')
    return f'<article><h3>{title}</h3>{"".join(rows)}</article>'


def count_html(label, count, previous):
    delta = 0 if previous is None else count - previous
    if delta:
        direction = 'up' if delta > 0 else 'down'
        change = f'<span class="delta-{direction}">{delta:+d}</span>'
    else:
        change = '<span class="pill">首次</span>' if previous is None else '<span class="pill">持平</span>'
    return f'<span>{label} <strong>{count}</strong> 篇 · {change}</span>'


def searches_html(searches, evidence, gold):
    rows = []
    previous_evidence = previous_gold = None
    for search in searches:
        query = search.get('query_text', search.get('query'))
        if query is None:
            query = search.get('tool_name', '查询未解析')
        if not isinstance(query, str):
            query = json.dumps(query, ensure_ascii=False)
        heading = f'<strong class="round-query">第 {search["round"]} 轮 · {esc(query)}</strong>'
        if search.get('docids') is None:
            rows.append(f'<li>{heading}<span class="pill">未完成有效检索</span></li>')
            continue
        docids = set(search['docids'])
        evidence_count, gold_count = len(docids & evidence), len(docids & gold)
        counts = (count_html('证据', evidence_count, previous_evidence)
                  + count_html('金文档', gold_count, previous_gold))
        rows.append(f'<li>{heading}<div class="round-counts">{counts}</div></li>')
        previous_evidence, previous_gold = evidence_count, gold_count
    return f'<ul class="rounds">{"".join(rows)}</ul>' if rows else '<p class="pill">无检索记录</p>'


def method_detail(method, record, qid, gold):
    if record is None:
        return f'<article><h3>{LABELS[method]}</h3><p>待运行</p></article>'
    evidence = set(record['evidence_docids'])
    hit = len(record['matched_evidence_docids'])
    hits = ', '.join(record['matched_evidence_docids']) or '—'
    first = record['first_evidence_round'] or '—'
    metrics = ' · '.join(f'{title} {show(metric(record, key), kind)}' for title, key, kind in METRICS[1:])
    searches = searches_html(record['searches'], evidence, gold)
    calls = []
    for number, call in enumerate(record['model_calls'], 1):
        stage = '收尾' if call['final_only'] else '常规'
        calls.append(f'<tr><td>{number}</td><td>{stage}</td><td>{show(call["prompt_tokens"], "count")}</td>'
                     f'<td>{show(call["completion_tokens"], "count")}</td>'
                     f'<td>{show(call["elapsed_seconds"], "seconds")}</td><td>{esc(call["finish_reason"])}</td></tr>')
    call_table = ('<details><summary>逐次模型调用</summary><div class="scroll"><table>'
                  '<thead><tr><th>序号</th><th>阶段</th><th>输入 token</th><th>输出 token</th>'
                  f'<th>耗时</th><th>结束原因</th></tr></thead><tbody>{"".join(calls)}</tbody></table></div></details>')
    error = f'<pre>{esc(record["error"])}</pre>' if record.get('error') else ''
    return (f'<article><h3>{LABELS[method]}</h3><p>证据 {hit}/{len(evidence)} · 首次命中轮次 {first}'
            f' · 尝试 {record["search_attempts"]} 次</p><p>{metrics}</p>'
            f'<p>检索 {show(record["retrieval_seconds"], "seconds")} · 重排 {show(record["rerank_seconds"], "seconds")}'
            f' · 模型请求 {show(record["model_seconds"], "seconds")}</p>'
            f'<p>已命中证据 docid：{esc(hits)} · 结束原因：{esc(record["termination"])}</p>'
            f'<a href="../runs/{qid}/{method}.json">原始 JSON</a>{error}'
            f'<h4>最终答案</h4><pre>{esc(record["prediction"])}</pre>'
            '<h4>逐轮召回</h4><p class="pill">统计当轮返回结果的命中数，增减相对上一轮有效检索。</p>'
            f'{searches}{call_table}</article>')


def generate_report(output):
    manifest = read_json(output / 'manifest.json')
    methods = manifest['methods']
    questions = dict(line.split('\t', 1) for line in
                     (output / 'queries.tsv').read_text(encoding='utf-8').splitlines())
    gold_labels = load_qrels(OFFICIAL / 'topics-qrels' / QRELS_FILES['gold'])
    if missing := questions.keys() - gold_labels.keys():
        raise ValueError(f'Missing gold judgments for questions: {", ".join(sorted(missing))}')
    by_method = {method: [] for method in methods}
    table_rows = []
    for qid in manifest['question_ids']:
        gold = {docid for docid, grade in gold_labels[qid].items() if grade > 0}
        details, cells = [], []
        for method in methods:
            path = run_path(output, qid, method)
            record = read_json(path) if path.is_file() else None
            if record and record['termination'] not in SERVICE_ERRORS:
                by_method[method].append(record)
                recall = show(record['evidence_recall'], 'percent')
                cells.append(f'<td>{recall} · {record["valid_searches"]} 次</td>')
            else:
                cells.append('<td>服务错误</td>' if record else '<td>待运行</td>')
            details.append(method_detail(method, record, qid, gold))
        heading = f'BrowseComp-Plus · 第 {qid} 题'
        content = (f'<p><a href="../report.html">← 返回总览</a></p><h1>{heading}</h1>'
                   f'<div class="box">{esc(questions[qid])}</div><h2>四组详情</h2>'
                   f'<div class="grid">{"".join(details)}</div>')
        atomic_text(output / 'questions' / f'{qid}.html', html_page(heading, content))
        table_rows.append(f'<tr><th><a href="questions/{qid}.html">{qid}</a></th><td>{esc(questions[qid])}'
                          f'</td>{"".join(cells)}</tr>')

    overview, summary = [], []
    for title, key, kind in METRICS:
        values = {method: [metric(record, key) for record in records if metric(record, key) is not None]
                  for method, records in by_method.items()}
        overview.append(chart(f'{title}均值', {method: mean(rows) if rows else None
                                             for method, rows in values.items()}, kind))
        stats = ''.join(f'<td>{show(mean(rows), kind)} / {show(median(rows), kind)}'
                        f' <span class="pill">n={len(rows)}</span></td>' if rows else '<td>—</td>'
                        for rows in values.values())
        summary.append(f'<tr><th>{title}</th>{stats}</tr>')
    total = sum(map(len, by_method.values()))
    expected = len(questions) * len(methods)
    endings = Counter(record['termination'] for rows in by_method.values() for record in rows)
    budget_answers = endings['answer at call limit'] + endings['answer at context limit']
    no_answers = total - budget_answers - endings['answer']
    headers = ''.join(f'<th>{LABELS[method]}</th>' for method in methods)
    title = 'BrowseComp-Plus · Tongyi 四组 Agent 实验'
    settings = manifest['settings']
    page = (f'<h1>{title}</h1><p><strong>{total}/{expected}</strong> 次正式运行已记录 · '
            f'{len(questions)} 道固定题 · 抽样种子 {manifest["sample_seed"]}</p>'
            '<p class="muted">证据召回按交给 Tongyi 的去重网页计算。搜索次数为有效搜索，'
            'token 为模型逐次调用的累计用量。答案尚未经过 Judge。样本较小，数值差异作描述性比较。</p>'
            f'<p>作答状态 · 正常作答 {endings["answer"]} 次 · 触发预算后作答 {budget_answers} 次'
            f' · 未提取到答案 {no_answers} 次</p>'
            f'<p class="muted">每轮前 {settings["top_k"]} 篇，每篇 {settings["snippet_max_tokens"]} tokens · '
            f'重排深度 {settings["rerank_depth"]} · 每题调用上限 {settings["max_llm_calls"]} · '
            '<a href="queries.tsv">固定题单</a> · <a href="manifest.json">完整配置</a></p>'
            f'<h2>四组指标</h2><div class="grid">{"".join(overview)}</div>'
            f'<div class="box scroll"><table><thead><tr><th>均值 / 中位数</th>{headers}</tr></thead>'
            f'<tbody>{"".join(summary)}</tbody></table></div>'
            '<h2>逐题比较</h2><p class="muted">每格显示证据召回和有效搜索次数。点题号查看答案和逐轮召回。</p>'
            '<input id="filter" type="search" aria-label="按题号或题目筛选" placeholder="按题号或题目筛选">'
            f'<div class="box scroll"><table><thead><tr><th>题号</th><th>题目</th>{headers}</tr></thead>'
            f'<tbody id="questions">{"".join(table_rows)}</tbody></table></div>'
            '<script>const input=document.querySelector("#filter"); const rows=document.querySelectorAll("#questions tr");'
            'input.addEventListener("input",()=>{const q=input.value.trim().toLowerCase();'
            'for(const row of rows){row.hidden=!row.textContent.toLowerCase().includes(q);}});</script>')
    atomic_text(output / 'report.html', html_page(title, page))
