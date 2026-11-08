"""Evidence and readable explanations for frozen nDCG@10 reranking regressions."""
from collections import Counter
import html
import math
from pathlib import Path

from .dataset import load_documents, load_qrels
from .schemas import LABELS
from .utils import digest, jsonl, object_hash, read_json


def extract_cases(run):
    run = Path(run)
    cfg = read_json(run / 'config.json')
    assert cfg['evaluation']['ndcg_scope'] == 'corpus'
    assert cfg['evaluation']['gain'] == 'linear'
    docs = {d.doc_id: d for d in load_documents(run / 'corpus.jsonl')}
    labels = load_qrels(run / 'qrels.tsv')
    metrics = read_json(run / 'metrics.json')['per_query']
    cases = []
    for record in jsonl(run / 'retrieval.jsonl'):
        q = record['qid']
        qlabels = labels.get(q, {})
        ideal = sorted([g for d, g in qlabels.items() if d in docs and g > 0], reverse=True)[:10]
        denominator = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        for base in ('bm25', 'bge'):
            other = base + '_rerank'
            before = metrics[base][q]['ndcg@10']
            after = metrics[other][q]['ndcg@10']
            if before is None or after is None or after - before >= -1e-12:
                continue
            old = {h['doc_id']: h for h in record['methods'][base]['hits']}
            new = {h['doc_id']: h for h in record['methods'][other]['hits']}
            assert set(old) == set(new)
            changes = []
            for doc_id in set(d for d, h in old.items() if h['rank'] <= 10) | set(d for d, h in new.items() if h['rank'] <= 10):
                d, grade = docs[doc_id], qlabels.get(doc_id)
                g = max(0, grade or 0)
                a, b = old[doc_id]['rank'], new[doc_id]['rank']
                score = lambda rank: g / math.log2(rank + 1) / denominator if rank <= 10 else 0
                changes.append({'doc_id': doc_id, 'title': d.title, 'text': d.text,
                                'search_text': d.search_text, 'grade': grade,
                                'before_rank': a, 'after_rank': b,
                                'before_score': old[doc_id]['score'], 'after_score': new[doc_id]['score'],
                                'ndcg_contribution_delta': score(b) - score(a)})
            changes.sort(key=lambda d: (d['ndcg_contribution_delta'], d['doc_id']))
            assert abs(sum(d['ndcg_contribution_delta'] for d in changes) - (after - before)) < 1e-12
            seq = lambda mapping: [qlabels.get(d) for d, h in sorted(mapping.items(), key=lambda pair: pair[1]['rank'])[:10]]
            cases.append({'key': base + '|' + q, 'base': base, 'qid': q, 'query': record['query'],
                          'before': before, 'after': after, 'delta': after - before,
                          'ideal_dcg_at_10': denominator, 'before_grades': seq(old), 'after_grades': seq(new),
                          'detail_file': 'q-' + object_hash(q)[:16] + '.html', 'changes': changes})
    cases.sort(key=lambda c: (c['base'], c['delta'], c['qid']))
    return cases


def provenance(run):
    run = Path(run)
    return {name: digest(run / name) for name in ('retrieval.jsonl', 'corpus.jsonl', 'qrels.tsv', 'config.json')}


def render_drop_panel(run):
    run = Path(run)
    data = read_json(run / 'rerank_drop10.json')
    if data['provenance'] != provenance(run):
        raise ValueError('退步分析与当前数据或配置不一致，请重新执行 tools/analyze_rerank_drops.py --extract --build。')
    if data['metric'] != 'ndcg@10' or data['ndcg_scope'] != 'corpus':
        raise ValueError('退步分析必须使用子集口径的nDCG@10。')
    expected = {c['key']: c for c in extract_cases(run)}
    if set(expected) != {c['key'] for c in data['cases']}:
        raise ValueError('退步分析存在遗漏或额外案例，请重建。')
    for c in data['cases']:
        if any(abs(c[k] - expected[c['key']][k]) > 1e-12 for k in ('before', 'after', 'delta')):
            raise ValueError('退步分析分数已过期，请重建。')
    esc = lambda value: html.escape(str(value), quote=True)
    signed = lambda value: '0.0000' if abs(value) <= 1e-12 else f'{value:+.4f}'
    grade = lambda value: '未标注' if value is None else str(value)
    categories = data['categories']
    target = lambda base, qid: 'drop-' + base + '-' + qid
    doc_link = lambda c, d, method: c['detail_file'] + '#doc-' + object_hash([method, d['doc_id']])[:16]
    output = ('<section class="panel" id="rerank-drop-analysis"><h2>重排退步案例与原因分类 · nDCG@10</h2>'
              f'<p>仅分析重排后nDCG@10下降的问题：BM25共{data["groups"]["bm25"]["count"]}题，BGE共{data["groups"]["bge"]["count"]}题。同一道题可在两组分别出现。'
              '使用固定1万段落内的同一理想排序；下降按未四舍五入分数判断。</p>'
              '<div class="notice"><strong>如何读“原因”：</strong>分数、位次、等级、原文和输入边界是可核对事实；'
              '对模型为何这样打分的解释主要是原文对照支持的推断。UMBRELA是自动标注，未标注不等于0级，'
              '高等级也不等于内容已获事实验证。本栏只分析退步案例，整体效果请结合上方两两方案比较。</div>'
              '<div class="toolbar"><label for="drop-method">检索器</label><select id="drop-method">'
              f'<option value="bm25">BM25：{data["groups"]["bm25"]["count"]}题</option><option value="bge">BGE：{data["groups"]["bge"]["count"]}题</option></select>'
              '<label for="drop-category">主原因类别</label><select id="drop-category"><option value="all">全部类别</option>')
    for key, info in categories.items():
        if any(c['category'] == key for c in data['cases']):
            output += f'<option value="{esc(key)}">{esc(info["title"])}</option>'
    output += ('</select><input id="drop-search" type="search" placeholder="搜索问题ID或关键词" '
               'aria-label="搜索退步案例"></div>')
    for method in ('bm25', 'bge'):
        cases = [c for c in data['cases'] if c['base'] == method]
        group = data['groups'][method]
        counts = Counter(c['category'] for c in cases)
        if len(cases) != group['count'] or dict(counts) != group['category_counts']:
            raise ValueError('原因分类统计与逐题清单不一致。')
        output += (f'<div class="drop-group" data-drop-method="{method}"{" hidden" if method != "bm25" else ""}>'
                   f'<h3>{esc(LABELS[method])} → {esc(LABELS[method + "_rerank"])}：{len(cases)}题下降</h3>'
                   '<p class="metricnote">每题只指定一个主类，次要标签不重复计数；下表题数对应本组全部下降题。'
                   '点击类别可筛选，点击问题ID可展开该题分析。</p>'
                   '<table class="drop-categories"><thead><tr><th>主类别</th><th>题数</th><th>对应问题ID</th></tr></thead><tbody>')
        for key, count in counts.most_common():
            info = categories[key]
            links = ' · '.join(f'<a class="drop-case-link" href="#{target(method,c["qid"])}">{esc(c["qid"])}</a>'
                               for c in cases if c['category'] == key)
            output += (f'<tr><td><button type="button" class="drop-category-button" data-drop-category="{esc(key)}">'
                       f'{esc(info["title"])}</button><div class="metricnote">{esc(info["description"])}</div></td>'
                       f'<td>{count}</td><td>{links}</td></tr>')
        output += '</tbody></table>'
        if counts:
            top, count = counts.most_common(1)[0]
            output += (f'<p><strong>本组总结：</strong>在这{len(cases)}个退步案例中，最多的主类是“{esc(categories[top]["title"])}”，'
                       f'共{count}题（{count/len(cases)*100:.1f}%）。这是退步样本内部的构成，不代表全部100题的原因分布。'
                       '不同解释可能同时成立，主类仅记录本次最有依据的主要线索。</p>')
        secondary_counts = Counter(tag for c in cases for tag in c['secondary'])
        if secondary_counts:
            output += '<details><summary>次要线索对应哪些题（允许重叠，不计入主类题数）</summary><table><thead><tr><th>次要线索</th><th>题数</th><th>问题ID</th></tr></thead><tbody>'
            for key, count in secondary_counts.most_common():
                links = ' · '.join(f'<a class="drop-case-link" href="#{target(method,c["qid"])}">{esc(c["qid"])}</a>'
                                   for c in cases if key in c['secondary'])
                output += f'<tr><td>{esc(categories[key]["title"])}</td><td>{count}</td><td style="white-space:normal">{links}</td></tr>'
            output += '</tbody></table></details>'
        output += '<p class="drop-visible"></p>'
        for c in cases:
            cid = target(method, c['qid'])
            search = (c['qid'] + ' ' + c['query'] + ' ' + c['title']).lower()
            output += (f'<details class="drop-case" id="{cid}" data-drop-category="{esc(c["category"])}" '
                       f'data-drop-search="{esc(search)}"><summary>'
                       f'<span class="drop-case-title">{esc(c["title"])} <small>{esc(c["qid"])}</small></span>'
                       f'<span>{c["before"]:.4f} → {c["after"]:.4f} '
                       f'<strong class="drop-negative">{signed(c["delta"])}</strong></span></summary>'
                       f'<div class="drop-body"><p><strong>原问题：</strong>{esc(c["query"])}</p>'
                       f'<p><strong>需要满足：</strong>{esc(c["intent"])}</p>'
                       f'<p><span class="badge">主类：{esc(categories[c["category"]]["title"])}</span>')
            output += ''.join(f'<span class="badge">次要：{esc(categories[tag]["title"])}</span>' for tag in c['secondary'])
            output += ('</p><h4>已核对：排名与计分变化</h4><div class="drop-grade-lines">')
            for title, values in [('重排前第1～10名', c['before_grades']), ('重排后第1～10名', c['after_grades'])]:
                output += f'<div>{title}：' + ''.join(f'<span class="badge {"unjudged" if g is None else "relevant" if g>0 else "nonrelevant"}">{grade(g)}</span>' for g in values) + '</div>'
            losses = [d for d in c['changes'] if d['ndcg_contribution_delta'] < -1e-12][:3]
            promotions = sorted([d for d in c['changes'] if d['after_rank'] < d['before_rank'] and d['after_rank'] <= 5], key=lambda d:d['after_rank'])[:2]
            improvements = sorted([d for d in c['changes'] if d['ndcg_contribution_delta'] > 1e-12], key=lambda d:-d['ndcg_contribution_delta'])[:1]
            selected = {d['doc_id']: d for d in losses + promotions + improvements}
            output += ('</div><p class="metricnote">下表选出主要扣分段落、前列新提升段落和主要增益段落。'
                       '正负影响共同相加才是最终变化；0级或未标注虽自身收益为0，仍可能占位挤掉正等级段落。</p>')
            def table(rows):
                result = '<div class="drop-table-scroll"><table><thead><tr><th>段落 / 标题</th><th>本题等级</th><th>重排前</th><th>重排后</th><th>对nDCG@10变化的贡献</th></tr></thead><tbody>'
                for d in rows:
                    result += (f'<tr><td class="drop-doc-title">{esc(d["title"] or "无标题段落")}<div class="docid">{esc(d["doc_id"])}</div></td>'
                               f'<td>{grade(d["grade"])}</td><td><a href="{doc_link(c,d,method)}">#{d["before_rank"]}</a></td>'
                               f'<td><a href="{doc_link(c,d,method+"_rerank")}">#{d["after_rank"]}</a></td>'
                               f'<td>{signed(d["ndcg_contribution_delta"])}</td></tr>')
                return result + '</tbody></table></div>'
            output += table(list(selected.values()))
            output += (f'<h4>原文对照</h4><p>{esc(c["reading"])}</p>'
                       f'<h4>下降原因：有依据的解释</h4><p>{esc(c["explanation"])}</p>'
                       f'<p class="metricnote">证据程度：{esc(c["confidence"])}</p>'
                       f'<p><strong>下一步怎样验证：</strong>{esc(c["next_check"])}</p>'
                       f'<p><a href="{c["detail_file"]}">打开本题完整检索结果 →</a></p>'
                       '<details><summary>展开关键原文与输入边界</summary>')
            truncated = [d for d in c['changes'] if d['tokenization']['truncated']]
            output += (f'<p>本案例前10变化涉及{len(c["changes"])}个段落，其中{len(truncated)}个输入发生截断。'
                       '以下长度按重排器实际的“问题＋标题和正文”分词方式核对；发生截断不等于已证明它造成退步。</p>')
            for d in selected.values():
                t = d['tokenization']
                output += (f'<article class="drop-excerpt"><h4>{esc(d["title"] or "无标题段落")} · 相关等级：{grade(d["grade"])}</h4>'
                           f'<p class="metricnote">完整配对 {t["full_pair_tokens"]} tokens；实际输入 {t["input_tokens"]} tokens；'
                           f'{"发生截断" if t["truncated"] else "未截断"}。</p>'
                           f'<p>{esc(d["text"][:650])}{"…" if len(d["text"])>650 else ""}</p>')
                if t['truncated']:
                    cut = t['retained_text_end']
                    output += (f'<p><strong>保留部分结尾：</strong>{esc(d["search_text"][max(0,cut-130):cut])}</p>'
                               f'<p><strong>被省略部分开头：</strong>{esc(d["search_text"][cut:cut+220])}</p>')
                output += f'<a href="{doc_link(c,d,method+"_rerank")}">查看完整原文及来源</a></article>'
            output += ('</details><details><summary>展开全部逐段贡献，核对总下降</summary>'
                       '<p>只统计两次前10并集。某段在前10外时贡献为0；其变化等于新旧位置折扣收益之差，再除以本题固定IDCG。'
                       '下表所有贡献之和，应等于本题总变化。</p>')
            output += table(c['changes'])
            output += (f'<p>贡献合计：{sum(d["ndcg_contribution_delta"] for d in c["changes"]):+.8f}；'
                       f'实际nDCG变化：{c["delta"]:+.8f}。</p></details></div></details>')
        output += '</div>'
    checks = data['input_checks']
    output += (f'<p class="metricnote">输入检查覆盖{checks["question_document_pairs"]}个不同“问题—段落”组合，'
               f'其中{checks["truncated_pairs"]}个超过输入上限。这里检查的是本次退步案例两次前10的并集，'
               '不是所有100候选。分类汇总只按主类计数，次标签仅提供额外检查方向。</p></section>')
    return output
