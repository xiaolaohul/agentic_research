"""Offline, per-query comparisons using a shared nDCG evaluation scope."""
import html
from pathlib import PurePosixPath

from .schemas import LABELS


def pair_comparison_panel(analysis):
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value: 'N/A' if value is None else f'{value:.4f}'
    signed = lambda value: 'N/A' if value is None else ('0.0000' if abs(value) <= 1e-12 else f'{value:+.4f}')
    comparisons = analysis['comparisons']
    # Keep a common, symmetric axis for all six groups, including all-tie data.
    scale = max((abs(row['delta']) for pair in comparisons.values() for row in pair['cases']), default=0) or 1
    content = (f'<section class="panel" id="pair-comparison"><h2>两两方案比较 · {esc(analysis["metric"])}</h2>'
               '<p>每道题的差值＝B方案得分－A方案得分。正数表示B更好，负数表示A更好。'
               '平均分和差值仅使用两种方案都有有效得分的同一批问题。</p>'
               '<div class="toolbar"><label for="pair-select">比较组合</label><select id="pair-select">')
    for index, (key, pair) in enumerate(comparisons.items()):
        content += (f'<option value="{esc(key)}"{" selected" if index == 0 else ""}>'
                    f'A：{esc(LABELS[pair["base"]])} → B：{esc(LABELS[pair["other"]])}</option>')
    content += ('</select><label for="pair-filter">问题范围</label><select id="pair-filter">'
                '<option value="all">全部问题</option><option value="win">B胜出</option>'
                '<option value="tie">持平</option><option value="loss">B落后</option></select>'
                '<label for="pair-sort">排列方式</label><select id="pair-sort">'
                '<option value="delta">差值从高到低</option><option value="absolute">变化幅度从大到小</option>'
                '<option value="qid">问题ID</option></select></div>')
    for index, (key, pair) in enumerate(comparisons.items()):
        base, other = LABELS[pair['base']], LABELS[pair['other']]
        content += (f'<div class="pair-view" data-pair="{esc(key)}"{" hidden" if index else ""}>'
                    '<div class="stats">')
        for label, value in [(f'A · {base} 平均分', fmt(pair['base_mean'])),
                             (f'B · {other} 平均分', fmt(pair['other_mean'])),
                             ('平均差值 B－A', signed(pair['mean_delta'])),
                             ('中位数差值 B－A', signed(pair['median_delta']))]:
            content += f'<div class="stat"><span class="muted">{esc(label)}</span><strong>{esc(value)}</strong></div>'
        content += (f'</div><div class="pair-counts"><span class="pair-positive">B胜出 {pair["wins"]} 题</span>'
                    f'<span>持平 {pair["ties"]} 题</span><span class="pair-negative">B落后 {pair["losses"]} 题</span>'
                    f'<span class="muted">有效配对 {pair["valid_queries"]} 题；缺少有效得分 {pair["excluded_queries"]} 题</span></div>'
                    f'<p class="metricnote">逐题差值图：← B落后 · 0 · B胜出 →；各比较组合统一范围为±{scale:.4f}。'
                    '点击任意一行进入该题的完整方案详情。胜平负根据未四舍五入的分数统计。</p>'
                    f'<p class="pair-visible">显示 {pair["valid_queries"]}/{pair["valid_queries"]} 题</p>'
                    '<div class="pair-chart"><div class="pair-head"><span>问题</span><span>差值（中心为0）</span>'
                    '<span>A得分</span><span>B得分</span><span>B－A</span><span>结果</span></div><div class="pair-rows">')
        for row in pair['cases']:
            delta = row['delta']
            status = 'win' if delta > 1e-12 else 'loss' if delta < -1e-12 else 'tie'
            caption = {'win': 'B胜', 'loss': 'B负', 'tie': '持平'}[status]
            width = abs(delta) / scale * 50 if status != 'tie' else 0
            left = 50 - width if status == 'loss' else 50
            filename = PurePosixPath(row['report']).name
            title = f'{row["query"]} | A={row["base_value"]:.8f} B={row["other_value"]:.8f} B－A={delta:+.8f}'
            content += (f'<a class="pair-row pair-{status}" data-pair-row data-qid="{esc(row["qid"])}" '
                        f'data-delta="{delta:.17g}" data-status="{status}" href="{esc(filename)}" title="{esc(title)}">'
                        f'<span class="pair-question"><span>{esc(row["query"])}</span><small>{esc(row["qid"])}</small></span>'
                        f'<span class="pair-track" aria-hidden="true"><span class="pair-zero"></span>'
                        f'<span class="pair-bar" style="left:{left:.6f}%;width:{width:.6f}%"></span></span>'
                        f'<span>{fmt(row["base_value"])}</span><span>{fmt(row["other_value"])}</span>'
                        f'<strong>{signed(delta)}</strong><span>{caption}</span></a>')
        if not pair['cases']:
            content += '<p class="muted">没有可比较的有效问题。</p>'
        content += '</div></div></div>'
    content += ('<p class="metricnote">平均差反映整体变化，中位数差反映中间位置问题的变化；结合胜平负和逐题图判断提升是否集中在少数问题。'
                '本栏是描述统计，不是统计显著性检验。</p></section>')
    return content
