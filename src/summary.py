"""A short, data-derived experiment note alongside the offline HTML report."""
from pathlib import Path
from statistics import mean

from .schemas import METHODS, LABELS
from .evaluation import ndcg_description
from .utils import atomic_text, jsonl, read_json


def number(value):
    return 'N/A' if value is None else f'{value:.4f}'


def write_summary(run, comparisons):
    run = Path(run)
    cfg, dataset, state, metrics = (read_json(run / f) for f in ('config.json', 'dataset.json', 'run.json', 'metrics.json'))
    METHODS = tuple(metrics['aggregate'])
    answers = list(jsonl(run / 'answers.jsonl')) if (run / 'answers.jsonl').exists() else []
    retrieval_only = state.get('report_mode') == 'retrieval_only'
    judged_report = state.get('report_profile') == 'retrieval_judged10'
    featured = 10 if 10 in cfg['evaluation']['ks'] else cfg['evaluation']['ks'][0]
    retrieval = cfg['retrieval']
    lines = [f'# 实验 {state["run_id"]}', '', dataset['warning'], '',
        f'- 语料：{dataset["documents"]} 个段落；问题：{dataset["queries"]} 个；随机种子：{dataset["seed"]}。',
        f'- 标注：{dataset["judgments"]}。',
        (f'- 候选数：{retrieval["candidate_k"]}；详情默认展示前{retrieval["display_k"]}条，可展开全部候选。' if retrieval_only else f'- 候选数：{retrieval["candidate_k"]}；生成证据数：{retrieval["context_k"]}。'),
        ('- 模式：仅检索评测；不进行答案生成。' if retrieval_only else f'- 生成状态：{state["generation_status"]}；已有答案：{len(answers)}/{dataset["queries"] * len(METHODS)}。'),
        '', '[打开 HTML 报告](report/index.html)', '', '## 检索结果', '',
        (f'| 方法 | nDCG@{featured} | MRR@{featured} | Judged@10 | 平均总耗时 ms |' if judged_report else
         f'| 方法 | nDCG@{featured} | Precision@5 | MRR@{featured} | Recall@{featured} | 平均总耗时 ms |'),
        ('|---|---:|---:|---:|---:|' if judged_report else '|---|---:|---:|---:|---:|---:|')]
    for method in METHODS:
        stats = metrics['aggregate'][method]
        keys = ((f'ndcg@{featured}', f'mrr@{featured}', 'judged@10', 'total_ms') if judged_report else
                (f'ndcg@{featured}', 'precision@5', f'mrr@{featured}', f'recall@{featured}', 'total_ms'))
        values = [stats[key]['mean'] for key in keys]
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(number(v) for v in values) + ' |')
    if not judged_report and 100 in cfg['evaluation']['ks']:
        lines += ['', '| 方法 | Recall@100，当前问题全部已标注相关段落作分母 |', '|---|---:|']
        for method in METHODS:
            stats = metrics['aggregate'][method]
            lines.append(f'| {LABELS[method]} | {number(stats["recall@100"]["mean"])} |')
    coverage = metrics['aggregate']['bm25']['corpus_relevant_coverage']['mean']
    if judged_report:
        lines += ['', 'Judged@10＝前10条中针对当前问题有标注记录的段落数÷10；等级0、1、2、3均计入，未标注和不足10条的空位不计入分子。总览为逐问题得分的平均值；有标注不等于相关。',
                  ndcg_description(cfg['evaluation'])]
    else:
        lines += ['', f'子集覆盖的相关段落比例，按问题平均：{number(coverage)}。',
            'Recall@K＝前K条中对当前问题相关的段落数÷原始标注中对当前问题相关的段落总数；相关阈值按实验配置，分母包含子集之外的相关段落，不包含等级0或未标注段落。',
            ndcg_description(cfg['evaluation']),
            f'Precision@5＝前5条中等级≥{cfg["evaluation"]["relevant_threshold"]}的段落数÷5；不足5条的空位计不相关。']
    lines += [
        '未标注在指标中计零增益；无相关正例的指标为 N/A。', '', '## 逐问题变化', '',
        f'比较指标：{comparisons["metric"]}。以下是描述统计，不是显著性检验。', '',
        '| 比较方向 | 上升 | 下降 | 持平 |', '|---|---:|---:|---:|']
    for pair in comparisons['comparisons'].values():
        lines.append(f'| {LABELS[pair["base"]]} → {LABELS[pair["other"]]} | {pair["wins"]} | {pair["losses"]} | {pair["ties"]} |')
    if state.get('show_rerank_drop_analysis', False):
        drops = read_json(run / 'rerank_drop10.json')
        lines += ['', '## 重排序退步案例 · nDCG@10', '',
                  '逐题原文对照、扣分贡献和分类清单见[HTML退步分析栏目](report/index.html#rerank-drop-analysis)。',
                  '排名、等级和计分变化是可核对事实；语义解释是有依据的推断，未标注不等于0级。', '',
                  '| 检索器 | 主类别 | 题数 | 对应问题 |', '|---|---|---:|---|']
        for method, group in drops['groups'].items():
            for category, questions in group['category_questions'].items():
                lines.append(f'| {LABELS[method]} | {drops["categories"][category]["title"]} | {len(questions)} | '+ '、'.join(questions) + ' |')
    lines += ['', '## 生成结果与复核', '', '| 方法 | 已完成 | 触及长度上限 | 无引用 | 含无效引用 ID | 平均秒数 |', '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        rows = [r for r in answers if r['method'] == method]
        seconds = mean(r['generation_ms'] / 1000 for r in rows) if rows else None
        lines.append(f'| {LABELS[method]} | {len(rows)} | {sum(r["output_truncated"] for r in rows)} | '
            f'{sum(not r["citations"] for r in rows)} | {sum(bool(r["invalid_citations"]) for r in rows)} | {number(seconds)} |')
    lines += ['', '无引用、无效引用和截断均单独标记，未自动修补。引用 ID 有效不等于事实正确或证据支持，答案质量仍需人工或专门评估。']
    if retrieval_only:
        section = lines.index('## 生成结果与复核')
        lines = lines[:section]
    if (run / 'performance.json').is_file() and not retrieval_only:
        performance = read_json(run / 'performance.json')
        lines += ['', '## 资源实测', '',
            f'- 整张显卡峰值：{performance.get("peak_gpu_used_mib")} MiB（包含桌面及其他程序）。',
            f'- 推理进程峰值工作集：{performance.get("peak_server_working_set_mib")} MiB。',
            f'- 最低可用系统内存：{performance.get("min_system_ram_available_mib")} MiB。',
            '固定间隔采样，可能漏掉短暂尖峰；详细记录见 performance.samples.jsonl。']
    if 'rrf' in METHODS:
        fusion = cfg['retrieval']['rrf']
        lines += ['', '## RRF与重排参数', '',
                  f'- 两路各取Top-{fusion["input_k"]}，等权RRF常数c={fusion["rank_constant"]}，输出Top-{fusion["output_k"]}。',
                  f'- BM25与BGE的模型重排深度均为{cfg["reranker"]["depth"]}，配对输入上限{cfg["reranker"]["max_length"]} tokens。',
                  '- 平均融合耗时：' + number(metrics['aggregate']['rrf']['fusion_ms']['mean']) + ' ms。',
                  state.get('timing_note', '两路检索顺序运行，另加融合耗时。')]
    provenance = dataset.get('provenance', {})
    publisher = provenance.get('publisher') or {}
    lines += ['', '## 来源与复现', '',
        f'- 类型：{provenance.get("kind", "unknown")}；抽样范围：{provenance.get("pool_scope", "N/A")}。',
        f'- 来源仓库：{publisher.get("repository", "N/A")}；Git 提交：{publisher.get("revision", "N/A")}。',
        f'- 原始正文 SHA256：{provenance.get("source_sha256", "N/A")}。',
        f'- 子集正文 SHA256：{dataset["corpus_sha256"]}。',
        f'- BGE：{cfg["retrieval"]["bge"]["model"]}，{state.get("bge_revision")}。',
        f'- Reranker：{cfg["reranker"]["model"]}，{state.get("reranker_revision")}。',
        '[配置快照](config.json) · [数据清单](dataset.json) · [逐问题指标](metrics.json) · [逐问题比较](paired_analysis.json)', '']
    if (run / 'metrics_validation.json').is_file():
        checked = read_json(run / 'metrics_validation.json')
        checks = ({k:v for k,v in checked['checks'].items() if k not in ('recall', 'precision')}
                  if judged_report else checked['checks'])
        lines.append(f'独立指标核对：{checked["provider"]}，通过={checked["passed"]}，核对数量={checks}。')
    atomic_text(run / 'summary.md', '\n'.join(lines) + '\n')
    return run / 'summary.md'
