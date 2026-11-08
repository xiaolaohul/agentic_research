from collections import Counter
import html
import json
from pathlib import Path
import re
from statistics import mean
from urllib.parse import urlparse

from .dataset import load_documents, load_qrels
from .schemas import methods_for, LABELS
from .utils import ROOT, atomic_text, jsonl, object_hash, read_json, write_json


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value):
    return 'N/A' if value is None else f'{value:.4f}'


def page(title, subtitle, content):
    template = (ROOT / 'templates/report.html').read_text(encoding='utf-8')
    # Replace the content marker last so untrusted text cannot expand template placeholders.
    document = template.replace('__TITLE__', esc(title)).replace('__SUBTITLE__', esc(subtitle)).replace('__CONTENT__', content)
    # Normalize source CRLF before Windows text-mode output, avoiding CRCRLF/extra blank lines.
    return document.replace('\r\n', '\n').replace('\r', '\n')


def answer_html(row):
    if not row:
        return '<div class="answer muted">尚未生成答案。检索实验可独立查看。</div>'
    text = esc(row['answer'])
    for source in row['evidence']:
        marker = '[' + esc(source['doc_id']) + ']'
        target = 'doc-' + object_hash([row['method'], source['doc_id']])[:16]
        text = text.replace(marker, f'<a href="#{target}">{marker}</a>')
    warning = '<div class="badge nonrelevant">输出达到长度上限</div>' if row['output_truncated'] else ''
    if row['invalid_citations']:
        warning += '<div class="badge nonrelevant">存在输入证据以外的引用</div>'
    speed = row.get('end_to_end_output_tokens_per_second')
    speed_text = f' · 端到端 {speed:.1f} tokens/s' if speed is not None else ''
    return (f'<div class="answer"><h3>Qwen 生成答案</h3>{text}{warning}'
            f'<div class="metricnote">生成 {row["generation_ms"] / 1000:.1f}s · '
            f'输入 {row["prompt_tokens_counted"]} tokens{speed_text} · 引用 ID 有效率 '
            f'{fmt(row["citation_id_precision"])}（不代表事实支撑性）</div>'
            f'<details><summary>实际输入证据与截断记录</summary>'
            f'<pre>{esc(json.dumps(row["evidence"], ensure_ascii=False, indent=2))}</pre></details></div>')


def generate_report(run, verify_metrics=False):
    run = Path(run)
    records = list(jsonl(run / 'retrieval.jsonl'))
    METHODS = methods_for(records)
    manifest, cfg = (read_json(run / f) for f in ('dataset.json', 'config.json'))
    state = read_json(run / 'run.json')
    retrieval_only = state.get('report_mode') == 'retrieval_only'
    judged_report = state.get('report_profile') == 'retrieval_judged10'
    docs = {d.doc_id: d for d in load_documents(run / 'corpus.jsonl')}
    labels = load_qrels(run / 'qrels.tsv')
    # Refresh old experiments from frozen rankings and labels, without model inference.
    from .evaluation import evaluate, ndcg_description
    from .pipeline import export_metrics_csv
    from .validation import paired_analysis, validate_metrics
    metrics = evaluate(records, labels, set(docs), cfg['evaluation'])
    write_json(run / 'metrics.json', metrics)
    export_metrics_csv(run, metrics)
    if verify_metrics:
        validate_metrics(run)
    show_pairs = state.get('show_pair_comparison', False)
    comparisons = paired_analysis(run, all_pairs=show_pairs)
    answers = {(r['qid'], r['method']): r for r in jsonl(run / 'answers.jsonl')} if (run / 'answers.jsonl').exists() and not retrieval_only else {}
    report = run / 'report'
    report.mkdir(exist_ok=True)
    ks = cfg['evaluation']['ks']
    featured = 10 if 10 in ks else ks[0]
    metric_key = f'ndcg@{featured}'
    secondary_key = 'judged@10' if judged_report else 'precision@5'
    secondary_label = 'Judged@10' if judged_report else 'Precision@5'
    ndcg_note = ndcg_description(cfg['evaluation'])
    overview_rows, filenames = [], {}
    for row in records:
        qid = row['qid']
        filename = 'q-' + object_hash(qid)[:16] + '.html'
        filenames[qid] = filename
        qlabels = labels.get(qid, {})
        shared = Counter(h['doc_id'] for method in METHODS for h in row['methods'][method]['hits'][:cfg['retrieval']['display_k']])
        columns = []
        for method in METHODS:
            data = row['methods'][method]
            qmetrics = metrics['per_query'][method][qid]
            body = (f'<section class="column"><div class="column-head"><h2>{LABELS[method]}</h2>'
                    f'<span class="pill">{len(data["hits"])} candidates</span>'
                    f'<div class="metricnote">nDCG@{featured} {fmt(qmetrics[metric_key])} · '
                    f'{secondary_label} {fmt(qmetrics[secondary_key])} · '
                    f'总耗时 {data["total_ms"]:.1f}ms</div></div>')
            if 'rerank_depth' in data and method.endswith('_rerank'):
                body += f'<p class="metricnote stage-note">重排深度 {data["rerank_depth"]} · 实际重排 {data["reranked_count"]} 条</p>'
            if method == 'rrf':
                body += f'<p class="metricnote stage-note">两路各Top-{cfg["retrieval"]["rrf"]["input_k"]} · 去重并集 {data["candidate_union_count"]} 条 · 融合 {data["fusion_ms"]:.3f}ms · 总耗时为串行估算</p>'
            if not retrieval_only:
                body += answer_html(answers.get((qid, method)))
            for i, hit in enumerate(data['hits']):
                doc = docs[hit['doc_id']]
                grade = qlabels.get(doc.doc_id)
                badge = ('unjudged', '未标注') if grade is None else (
                    ('relevant', f'相关性 {grade}') if grade >= cfg['evaluation']['relevant_threshold']
                    else ('nonrelevant', f'已标注 {grade}'))
                delta = hit['retrieval_rank'] - hit['rank']
                movement = '' if hit['rerank_score'] is None else (
                    f'<span class="change {"down" if delta < 0 else ""}">'
                    f'{"↑" if delta > 0 else "↓" if delta < 0 else "="}{abs(delta)}'
                    f'（原 #{hit["retrieval_rank"]}）</span>')
                common = f'<span class="badge shared">前列 {shared[doc.doc_id]} 组共有</span>' if shared[doc.doc_id] > 1 else ''
                url = f'<a href="{esc(doc.url)}" target="_blank" rel="noopener noreferrer">原始来源 ↗</a>' if urlparse(doc.url).scheme in ('http', 'https') else ''
                extra = ' extra-hit' if i >= cfg['retrieval']['display_k'] else ''
                target = 'doc-' + object_hash([method, doc.doc_id])[:16]
                rerank_score = '' if hit['rerank_score'] is None else f' · 重排 {hit["rerank_score"]:.4f}'
                fusion_note = ''
                if method == 'rrf':
                    fusion_note = (f'<div class="metricnote">RRF融合分 {hit["score"]:.8f} · BM25原名次 {hit["bm25_rank"] or "未入选"}'
                                   f' · BGE原名次 {hit["bge_rank"] or "未入选"}<br>'
                                   f'两路贡献：{hit["bm25_contribution"]:.8f} + {hit["bge_contribution"]:.8f}</div>')
                body += (f'<article class="hit{extra}" data-doc="{esc(doc.doc_id)}" id="{target}">'
                    f'<span class="rank">#{hit["rank"]}</span>{movement}'
                    f'<span class="badge {badge[0]}">{esc(badge[1])}</span>{common}'
                    f'<h3>{esc(doc.title or "无标题段落")}</h3><div class="docid">{esc(doc.doc_id)}</div>'
                    f'<div>{esc(doc.text[:230])}{"…" if len(doc.text) > 230 else ""}</div>'
                    + (fusion_note or f'<div class="metricnote">检索分 {hit["retrieval_score"]:.4f}{rerank_score}</div>')
                    + f'<details><summary>展开全文与来源</summary><div class="fulltext">{esc(doc.text)}</div>{url}</details></article>')
            columns.append(body + '</section>')
        nav = (f'<div class="links"><a href="index.html">← 实验总览</a>'
               f'<span class="muted">{esc(qid)}</span></div>')
        reference = metrics['per_query']['bm25'][qid]
        coverage_note = (f'本题子集内有 {reference["relevant_in_corpus"]} 个已标注相关段落；'
                         f'原始标注中共 {reference["relevant_total"]} 个。')
        content = nav + f'<div class="notice">{esc(manifest["warning"])}<br>{esc(coverage_note)}<br>{esc(ndcg_note)}</div>'
        content += ('<div class="toolbar"><button id="expand">显示所有候选</button>'
                    '<span class="muted">悬停段落可突出各组相同结果；不同模型分数不可直接横向比较。</span></div>')
        content += f'<div class="grid" style="--method-count:{len(METHODS)}">' + ''.join(columns) + '</div>'
        atomic_text(report / filename, page(row['query'], f'{qid} · {len(METHODS)}种检索方案并排对比', content))
        cells = ''.join(f'<td>{fmt(metrics["per_query"][m][qid][metric_key])}</td>' for m in METHODS)
        coverage = metrics['per_query']['bm25'][qid]['corpus_relevant_coverage']
        generated = [answers[(qid, m)] for m in METHODS if (qid, m) in answers]
        review_flags = []
        for flag, count in [('无引用', sum(not a['citations'] for a in generated)),
                            ('截断', sum(a['output_truncated'] for a in generated)),
                            ('无效 ID', sum(bool(a['invalid_citations']) for a in generated))]:
            if count:
                review_flags.append(f'{flag} {count} 组')
        review_text = ' / '.join(review_flags) if review_flags else ('—' if generated else '尚未生成')
        overview_rows.append(f'<tr data-query="{esc((qid + " " + row["query"]).lower())}">'
            f'<td><a class="question" href="{filename}">{esc(row["query"])}</a><div class="docid">{esc(qid)}</div></td>'
            + cells + ('' if judged_report else f'<td>{fmt(coverage)}</td>') + ('' if retrieval_only else f'<td>{esc(review_text)}</td>') + '</tr>')
    pool_note = ('<br>语料来自发布方多阶段 Top-100 候选池的固定抽样；抽样不使用相关性分数。'
                 '候选池已被其他模型筛选，不能据此推断完整语料上的效果。') if manifest.get('provenance', {}).get('kind') == 'candidate_pool' else ''
    content = f'<div class="notice"><strong>{esc(manifest["warning"])}</strong><br>标注来源：{esc(manifest["judgments"])}{pool_note}</div>'
    final_stat = ('检索评测记录', len(records)*len(METHODS)) if retrieval_only else ('答案生成', f'{len(answers)}/{len(records)*len(METHODS)}')
    content += '<div class="stats">' + ''.join(f'<div class="stat"><span class="muted">{label}</span><strong>{esc(value)}</strong></div>'
        for label, value in [('固定段落数', len(docs)), ('实验问题数', len(records)), ('检索方案', len(METHODS)),
                             final_stat]) + '</div>'
    if 'rrf' in METHODS:
        fusion = cfg['retrieval']['rrf']
        content += (f'<div class="panel" id="retrieval-settings"><h2>融合参数与重排深度</h2>'
                    f'<p>BM25和BGE各取前{fusion["input_k"]}条，按段落ID去重，等权RRF融合，'
                    f'常数c={fusion["rank_constant"]}，输出前{fusion["output_k"]}条。'
                    '融合分＝各路1/(c+原名次)之和；未入选某一路时该路贡献为0；同分按段落ID排序。</p>'
                    f'<p>BM25＋重排、BGE＋重排：深度均为{cfg["reranker"]["depth"]}，'
                    f'使用{esc(cfg["reranker"]["model"])}；配对输入上限{cfg["reranker"]["max_length"]} tokens。'
                    '重排深度是处理的候选条数，输入长度是每次问题与段落的token上限。</p>'
                    f'<p class="metricnote">{esc(state.get("timing_note", "两路检索顺序运行，另加融合耗时。"))}</p></div>')
    if (run / 'cohorts.json').is_file():
        from .cohorts import cohort_panel
        cohort_content = cohort_panel(run, records, labels, set(docs), cfg['evaluation'], judged_report=judged_report)
        if state.get('show_cohort_summary', True):
            content += cohort_content
    content += '<div class="panel"><h2>检索效果对比</h2><div class="toolbar"><span>选择指标</span><select id="metric">'
    plot_keys = ([f'{metric}@{k}' for metric in ('ndcg', 'mrr') for k in ks] + ['judged@10'] if judged_report else
                 [f'{metric}@{k}' for metric in ('ndcg', 'mrr', 'recall') for k in ks] + ['precision@5'])
    content += ''.join(f'<option value="{key}" {"selected" if key == metric_key else ""}>{esc(key.replace("judged", "Judged"))}</option>' for key in plot_keys) + '</select></div>'
    for method in METHODS:
        values = {key: metrics['aggregate'][method][key]['mean'] for key in plot_keys}
        content += (f'<div class="barrow" data-metrics="{esc(json.dumps(values))}"><span>{LABELS[method]}</span>'
                    '<div class="bartrack"><div class="bar"></div></div><strong class="barvalue"></strong></div>')
    if judged_report:
        content += (f'<p class="metricnote">{esc(ndcg_note)}<br>Judged@10＝前10条中，针对当前问题有标注记录的段落数÷10。'
                    '等级0、1、2、3均算有标注；未标注和不足10条的空位不计入分子，分母固定为10。'
                    '有标注不等于相关；该指标衡量标注覆盖程度，不衡量相关性高低。总览为逐问题得分的平均值。</p></div>')
    else:
        content += (f'<p class="metricnote">{esc(ndcg_note)}<br>Recall@K＝前K条中对当前问题相关的段落数÷原始标注中对当前问题相关的段落总数（包含语料子集之外的相关段落）。'
                '等级0和未标注段落不计入Recall分母；Precision@5＝前5个结果中相关段落数÷5。'
                f'相关等级≥{cfg["evaluation"]["relevant_threshold"]}视为相关，未标注和不足5条的空位计不相关。'
                'N/A 表示没有可用的相关正例。</p></div>')
    columns = ([f'ndcg@{featured}', f'mrr@{featured}', 'judged@10'] if judged_report else
               [f'ndcg@{featured}', 'precision@5', f'mrr@{featured}', f'recall@{featured}'])
    if not judged_report and 100 in ks and featured != 100:
        columns.append('recall@100')
    columns += ['retrieval_ms'] + (['fusion_ms'] if 'rrf' in METHODS else []) + ['rerank_ms', 'total_ms']
    content += '<div class="panel"><h2>指标与耗时</h2><table><thead><tr><th>方案</th>'
    content += ''.join(f'<th>{esc(key.replace("judged", "Judged"))}</th>' for key in columns) + '<th>有效评测问题</th></tr></thead><tbody>'
    for method in METHODS:
        stats = metrics['aggregate'][method]
        content += f'<tr><td>{LABELS[method]}</td>' + ''.join(f'<td>{fmt(stats.get(key, {"mean": 0.0})["mean"])}</td>' for key in columns)
        content += f'<td>{stats[metric_key]["valid_queries"]}/{len(records)}</td></tr>'
    content += '</tbody></table><p class="metricnote">毫秒 ms；模型加载不计入逐问题耗时，查询编码计入 BGE 检索耗时。</p></div>'
    if show_pairs:
        from .pair_report import pair_comparison_panel
        content += pair_comparison_panel(comparisons)
    else:
        content += f'<div class="panel"><h2>逐问题得失 · {esc(comparisons["metric"])}</h2><table><thead><tr><th>比较方向</th><th>上升</th><th>下降</th><th>持平</th><th>变化最大的问题</th></tr></thead><tbody>'
        for comparison in comparisons['comparisons'].values():
            cases = comparison['cases']
            links = []
            for case, caption in ([(cases[0], '最大上升'), (cases[-1], '最大下降')] if cases else []):
                filename = case['report'].split('/')[-1]
                links.append(f'<a href="{esc(filename)}">{caption} {esc(case["qid"])} ({case["delta"]:+.4f})</a>')
            content += (f'<tr><td>{LABELS[comparison["base"]]} → {LABELS[comparison["other"]]}</td>'
                f'<td>{comparison["wins"]}</td><td>{comparison["losses"]}</td><td>{comparison["ties"]}</td>'
                f'<td>{" · ".join(links)}</td></tr>')
        content += '</tbody></table><p class="metricnote">这里只描述本轮问题上的变化，不代表统计显著性或完整语料上的结论。</p></div>'
    if state.get('show_rerank_drop_analysis', False):
        from .rerank_drops import render_drop_panel
        content += render_drop_panel(run)
    if answers:
        content += '<div class="panel"><h2>各组答案生成</h2><table><thead><tr><th>方案</th><th>已完成答案</th><th>平均生成秒数</th><th>平均端到端 tokens/s</th><th>达到输出上限</th><th>无引用答案</th><th>无效引用 ID</th></tr></thead><tbody>'
        for method in METHODS:
            rows = [r for r in answers.values() if r['method'] == method]
            speeds = [r['end_to_end_output_tokens_per_second'] for r in rows if r.get('end_to_end_output_tokens_per_second') is not None]
            content += (f'<tr><td>{LABELS[method]}</td><td>{len(rows)}/{len(records)}</td>'
                f'<td>{fmt(mean(r["generation_ms"] / 1000 for r in rows) if rows else None)}</td>'
                f'<td>{fmt(mean(speeds) if speeds else None)}</td>'
                f'<td>{sum(r["output_truncated"] for r in rows)}</td><td>{sum(not r["citations"] for r in rows)}</td><td>{sum(len(r["invalid_citations"]) for r in rows)}</td></tr>')
        content += '</tbody></table><p class="metricnote">端到端速度含提示词处理和 HTTP 开销；引用 ID 检查不等于内容正确性评估。</p></div>'
    if (run / 'performance.json').is_file() and not retrieval_only:
        performance = read_json(run / 'performance.json')
        content += '<div class="panel"><h2>本机资源实测</h2><table><tbody>'
        for key, title in [('peak_gpu_used_mib', '整张显卡峰值占用 MiB'), ('baseline_gpu_used_mib', '启动前显卡占用 MiB'),
                           ('peak_server_working_set_mib', '推理服务峰值工作集 MiB'),
                           ('min_system_ram_available_mib', '最低可用系统内存 MiB')]:
            content += f'<tr><td>{title}</td><td>{fmt(performance.get(key))}</td></tr>'
        content += ('</tbody></table><p class="metricnote">显卡数值包含桌面及其他应用；按固定间隔采样，可能漏掉短暂尖峰。'
                    '只有项目自动启动的服务才记录进程内存。</p></div>')
    content += (f'<div class="panel"><h2>逐问题查看 · nDCG@{featured}</h2><div class="toolbar">'
                '<input id="search" type="search" placeholder="搜索问题文字或问题 ID" aria-label="搜索问题"></div>'
                '<table><thead><tr><th>问题</th>')
    content += ''.join(f'<th>{LABELS[m]}</th>' for m in METHODS) + ('' if judged_report else '<th>子集相关段落覆盖</th>') + ('' if retrieval_only else '<th>生成复核标记</th>') + '</tr></thead><tbody>' + ''.join(overview_rows) + '</tbody></table>' + ('' if retrieval_only else '<p class="metricnote">复核标记只检查引用格式与输出长度，不判断事实或语义正确性。</p>') + '</div>'
    content += '<div class="panel"><h2>评测口径与可复现信息</h2>'
    status_text = '仅检索评测，已完成' if retrieval_only else '生成状态：' + state['generation_status']
    content += f'<p>实验编号：<code>{esc(state["run_id"])}</code> · {esc(status_text)}</p>'
    content += '<p>未标注段落在指标中计零增益，在详情中单独显示。没有相关正例的问题不计入相应指标均值，并显示有效问题数。</p>'
    content += '<details><summary>数据来源、指纹与抽样方法</summary><pre>' + esc(json.dumps(manifest, ensure_ascii=False, indent=2)) + '</pre></details>'
    generation_info = read_json(run / 'generation.json') if (run / 'generation.json').is_file() and not retrieval_only else None
    displayed_cfg = {k:v for k,v in cfg.items() if k != 'generation'} if retrieval_only else cfg
    content += '<details><summary>配置与软件版本</summary><pre>' + esc(json.dumps({'retrieval_run_config': displayed_cfg, 'generation': generation_info, 'versions': state['versions'],
                'bge_revision': state.get('bge_revision'), 'reranker_revision': state.get('reranker_revision')}, ensure_ascii=False, indent=2)) + '</pre></details></div>'
    atomic_text(report / 'index.html', page(f'{len(METHODS)}种检索方案对比', f'{state["run_id"]} · 同一语料、同一问题、同一重排序模型', content))
    from .summary import write_summary
    write_summary(run, comparisons)
    print(f'HTML: {report / "index.html"}', flush=True)
    return report / 'index.html'
