"""Build a standalone, evidence-linked interpretation of the frozen TREC subset run."""
from pathlib import Path
from collections import Counter
from statistics import mean
import html
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import read_json, jsonl, object_hash, digest, atomic_text
from src.dataset import load_documents, load_qrels
from src.retrieval.bm25 import BM25Retriever
from src.schemas import METHODS, LABELS

RUN = ROOT / 'results/trec24-subset-20261027'
OUT = ROOT / 'results/fenxi.html'

# These are interpretations of the passages, not hidden-state explanations of a neural model.
# Fields: short title, intended constraints, BM25 reading, BGE reading, reranking reading, next check.
NOTES = {
 '2024-224063': (
  '为什么还会读写得不好的书',
  '要回答“明知写得不好，为什么仍然读”，而不是“阅读能力差的人有哪些困难”。',
  'BM25前两名是读者书评与阅读价值讨论，均为2级；但“如何写作”的教程也匹配 writing / books 等词，前10中仍有0级写作策略结果。',
  'BGE第1名讲 poor readers 的解码困难，标注为0级。它与“poorly written books”共享阅读困难的语境，却把“书写得差”偏移成“读者读得差”。这是正文可见的语义角色混淆。',
  '两组重排前5均变成4个1级和1个2级。Precision@5提高，但1级的“坏写作批评”排在能提供更多答案信息的2级之前，因此不保证nDCG提高。BGE与BM25候选重排后可高度接近，仍须核对前10是否完全相同。',
  '对照改写为“Why do readers continue reading books despite poor writing?”，检查读者/作品的角色是否被区分；这是待做的诊断，不应改写原正式成绩。'),
 '2024-22919': (
  '投资者是否会忽略市场噪声',
  '对象必须是投资者和金融市场，核心是是否忽略市场噪声及其理由。',
  'BM25优先取到投资建议与加密货币文章，前5均≥1级；但其中1级只解释“市场噪声”，不能等同于回答投资者是否同意忽略它。',
  'BGE第2名是公司文化口号“Ignore the noise”，标为0级。相同短语被放进了非投资决策的语境，说明句子相似并未自动满足金融主体限定。',
  '重排把公司口号移出前列，并让两组前10顺序一致；但首位仍是1级的市场信息介绍，所以两组MRR可很高，而nDCG不一定超过BM25直接排序。',
  '优先人工核对“噪声定义”和“支持/反对忽略噪声”的区别；若做查询改写，应明确金融投资决策语境。'),
 '2024-36155': (
  '墨西哥壁画如何影响社会结构',
  '必须连接“墨西哥壁画运动”与社会阶层、公共传播或社会变革的影响，不能只讲墨西哥政治史。',
  'BM25第3名是墨西哥政治与社会结构的地理题卡，标为0级。它匹配 Mexico / social structures，却未建立“壁画导致影响”的关系。',
  'BGE也把同一题卡排第3，但前两名是壁画运动的影响与遗产，均2级。它改善了主题聚焦，仍没有完全排除只匹配社会结构的干扰。',
  '重排移走题卡，并把原先在BGE第34名的3级段落提前到第1；两组前10一致。这里的提升是已有相关候选被重新组织，不是重排发现了新文档。',
  '检查每条证据是否写出了壁画与社会变化的联系，而非只同时提到墨西哥和社会。'),
 '2024-158677': (
  '1990年代美国娱乐发生了什么',
  '需要同时满足1990年代、美国与娱乐三个限定；泛泛谈名人或娱乐概念并不充分。',
  'BM25前5中有3条未标注，首条是名人文化/电视娱乐讨论。它未必实际不相关，但当前qrels不给它得分，不能把它直接称为0级。',
  'BGE更靠前召回了90年代娱乐回顾与当时家庭生活的段落，前5均有正等级；说明它在本题更好地聚合了年代与娱乐语境。',
  '重排把BM25原第8、6、49名提升到前3。两组重排的前5顺序确实相同，但前10不同；不同100候选池限制了可用材料。相同Precision@5不等于相同nDCG@10。',
  '特别查看第6～10名的差异，并把“未标注”与“已判不相关”分开；不要只依据前3名相同推断整组排序相同。'),
 '2024-38986': (
  '怎样支持其他星球上的生命',
  '“support life”有“维持生命条件”和“人类如何在外星生存”的歧义；需明确回答范围。',
  'BM25把宜居条件文章排在前列，前5全部≥1级；“如何发现行星”等背景也被标为1级，但不能直接充当生命支持方案。',
  'BGE前列有生命存在与太阳系概述，首两条仅1级；另有未标注结果。主题接近并不等于解释可行的生命支持条件。',
  '两组重排前10相同，优先提升“能否居住其他星球”的段落；首位为2级，但仍保留多个1级背景，故P@5为1也不代表5条均能直接回答。',
  '先固定任务是宜居条件还是工程生命支持，再人工评估证据是否满足同一个意图。'),
 '2024-43905': (
  '海湾国家怎样投资采矿',
  '按“海湾国家把资金投入采矿”的意图解读；Gulf states也可能被误解成美国墨西哥湾沿岸州。',
  'BM25第2名是墨西哥湾漏油赔偿，第3名讲吉布提矿业，均0级；它们匹配 Gulf / money / mining 的不同部分，却没同时满足投资主体、资金流向与采矿行业。',
  'BGE更偏向海湾阿拉伯国家经济背景，但吉布提矿业仍排第2且为0级。地理邻近、资源产业相似并不能代替“谁在投资谁”。',
  '两组重排都把阿富汗采矿收入与海湾私人资助的报道排第1，它只有1级：同段同时出现采矿和海湾资金，但二者不必构成题目要求的直接投资关系。后面0级经济背景仍需过滤。',
  '核对资金关系的主语、受益者和行业，不能把“段落里出现两个概念”当作两者确有直接联系。'),
 '2024-220140': (
  '英国40岁以下人群糖尿病问题',
  '限定是英国、40岁以下和原因解释；仅讨论全球糖尿病或别国“糖尿病之都”不够。',
  'BM25第2名是印度“diabetes capital of the world”，标为0级，属于高度重合短语掩盖了国家错误。',
  'BGE前两名为2级的英国相关段落，但同样把印度文章排第3；主题相似仍未完全约束国家和年龄。',
  '重排把一般英国统计与青年糖尿病文献提前，但前5变为3条1级和2条0级；两组都出现nDCG下降。更高的重排分不保证符合qrels对年龄与原因的要求。',
  '逐段检查是否真的涉及40岁以下与原因；网页健康陈述仅作为检索样本，不因等级高就视为医学事实已验证。'),
 '2024-158743': (
  '1840年代德国与荷兰的事件',
  '要同时关照1840年代和两个国家，泛年代事件表或只讲其他国家的移民题不能完整满足。',
  'BM25首位是欧洲移民回顾（2级），第2名却是选择题题干（0级）。大量 Germany / 1840s 共词，可能掩盖答案不完整。',
  'BGE前列更多是1840年代概览和1848革命，主题覆盖较广；仍会召回不含这两个国家答案的泛年代事件。',
  '两组都把“1840年发生了什么”的1级年表升为第1，压在2级德国革命段落前。单一年份与整个年代、全球与指定国家的粒度差异，是本题值得核对的错排线索。',
  '把“1840s”与“1840年”区分，并要求分别标记德国、荷兰的证据覆盖；不能仅凭题目式标题判断答案充分。'),
 '2024-94706': (
  '东西方安乐死观念有何差异',
  '题目要求东西方比较，单一宗教立场或安乐死历史只提供部分背景。',
  'BM25前5全是1级，前10没有0级。这不是“全不相关”，而是主题相关但比较答案不足；MRR/P@5会掩盖这种不足。',
  'BGE提前了包含东方宗教背景的2级段落，但也有1级综述与未标注条目，需检查是否真正形成东西方对照。',
  '两组重排前10一致，却让1级历史材料排第1，2级宗教比较材料排第2；未发现0级前排不代表排序已最优。',
  '人工标记每段能支撑的地域与观点，再看是否能组成对照；不要将“出现euthanasia”当作回答了比较问题。'),
 '2024-42645': (
  '特殊群体手册怎样帮助公平',
  '“special population handbook”指代不够明确；分析按教育特殊群体和公平的相关上下文解读，不能自行认定某一本手册。',
  'BM25将公共健康公平介绍排第1，特殊需求定义排第2且为0级，反映 equity / special population 共词并未锁定手册内容和教育场景。',
  'BGE首位是公共健康特殊人群地图，法律定义也进前列；它扩大了同义主题，却仍混合公共健康、法律与教育几种场景。',
  '重排提升Perkins职业教育与公平需求评估材料；BGE原第55、63名的2级结果升到第2、3。两组前10相同，说明关键材料原先都已召回，主要差在初始排序。',
  '优先确认手册全名、机构或使用场景；否则不能把一般“特殊群体”材料当作指定手册的回答。'),
 '2024-36302': (
  'Riaño居民怎样抗议搬迁',
  '需要同一实体Riaño及其居民反搬迁行动；其他地点的搬迁抗争不能替代。',
  'BM25前排是日裔美国人拘禁搬迁、美国原住民迁移等0级材料；共词residents / relocation / protest压过了目标实体限定。',
  'BGE同样偏向“群体抗议搬迁”的相似事件。库中有Riaño字符串的材料涉及其他历史人名，字符串出现也不等于命中同一实体。',
  '两组重排都把“营地居民通过法院挑战搬迁”的0级段落排第1：行为关系很像，但主体错误。另有一个Poletown/Detroit段落被qrels标为3级，与题面实体不一致，属于应人工复核的标注疑点，不能直接当作正确答案。',
  '先确认目标实体材料是否实际存在于子集；再复核跨城市的高等级标注。本题不宜只凭当前MRR判断真实任务已解决。'),
 '2024-42497': (
  '非正规电子垃圾焚烧怎样污染空气',
  '需要“非正规回收—焚烧电子垃圾—空气排放”的关系，普通回收减排或一般垃圾焚烧不是完整答案。',
  'BM25第2名“回收如何减少污染”被标0级；recycling / pollution共词较多，但因果方向与题意相反。一般塑料焚烧文章也仅1级。',
  'BGE前列更多为电子垃圾处理和露天拆解的3级段落，表明本题具体过程的语义召回比泛关键词更有效。',
  '两组重排把2～3级直接机制段落集中前排，前5均3级，且前10完全一致；0级“回收减排”被压到第8，仍影响更深截断的nDCG。',
  '按因果方向核对原文，区分“正规回收减少污染”和“非正规焚烧产生污染”；这些过程不能仅因共享关键词就混在一起。'),
 '2024-225223': (
  '为什么不应改变言论自由权利',
  '原题priviledges疑似拼写错误，且含“不应改变”的立场限定；支持限制与反对限制的文本都可能使用同样词语。',
  'BM25找到不少2/3级理由材料，但第7名是priviledge词典定义（0级），属于拼写/词面命中而缺少言论自由论证。',
  'BGE直接排序前列有3级的自由言论理由材料，也夹有讨论“应限制言论”的条目；同一争议话题的正反立场不能混为答案。',
  '重排前5均≥1级，但把含“should be changed”等措辞的混合立场文章升为首位，2/3级分布并未最优。它展示了P@5为1而nDCG下降的可能，不应只因主题正确就认为立场正确。',
  '人工复核否定词、主张立场和论据；“有争论”与“支持题目要求的理由”是不同层次。'),
 '2024-44048': (
  'Alpine Road沿线旅游如何变化',
  '道路名称缺少国家和明确实体，需区分阿尔卑斯地区旅游变化与其他同名/近名山路。',
  'BM25第3名是美国科罗拉多Alpine Loop，标为0级；同名山地道路造成地点漂移。首位环境与旅游文章仅1级，未必讲目标道路的变化。',
  'BGE更多召回阿尔卑斯旅游、滑雪和历史概览，前5没有0级；但地区旅游概述不能自动等于特定道路旅游演变。',
  '两组重排前10一致，移走科罗拉多道路并提升2级旅游材料，但首位仍是1级环境文章，因此nDCG改善并不意味着实体歧义已经解决。',
  '确认Alpine Road具体指哪条路，再区分地区背景、道路建设变化和游客活动变化。'),
 '2024-36355': (
  '1700年代怎样获得想要的香味（意图待确认）',
  '原文sceant不是常见规范拼写；按scent/香味理解只是诊断假设，不能代替原题。',
  '已有实测：原题1700\'s被分成1700，目标标题1700s保留为1700s，未匹配；唯一1级段落原排2798。只统一年份后升第39，故BM25前100漏召回有直接的预处理证据。',
  'BGE能把这个1级化妆品段落排第4，但它只谈胭脂、花与油，未直接解释香味；其余前列多为发型或1700年代生活。子集仅此1条正等级，不能据此声称已找到充分答案。',
  '已有单题对照：原题重排第6；只将sceant替换scent，保持100个候选不变，目标升第1。原顺序及分数完全复现，支持“该词影响排序”的结论，但不揭示模型内部思路。BM25未召回该段落，重排无法补回。',
  '正式原题保留；把日期规范化和拼写假设作为独立对照。两种问题分别出在第一阶段词匹配与第二阶段查询理解，不能混成同一原因。'),
 '2024-42781': (
  '把教学大纲变成交互平台',
  '原问题句式未完整说明“如何/产生什么效果”，可辨认约束是syllabus与interactive platform。',
  'BM25首位是Canvas课程中的syllabus功能（2级），但前5有未标注在线课程材料；第9名只是“怎样写教学大纲”（0级），缺少交互改造。',
  'BGE把PPT转交互课件、学习平台与互动讲座等材料排前，它们多为1级：接近交互教学，但对象未必是大纲本身。',
  '两组都把明确出现interactive syllabus和Thinglink的1级条目提前；Canvas的2级条目排第2。重排改善了主题贴合，仍不能把工具名命中等同于完整实现方案。',
  '先明确想问实施步骤还是学习效果，再检查证据是否真的以syllabus为对象。'),
 '2024-41918': (
  '钚如何用于技术改进',
  '必须是plutonium（钚）的技术应用，不能用名称近似的polonium（钋）或泛科技收益替代。',
  'BM25首位是钚的RTG应用段落（2级），但历史援助与工业生产材料未标注，可能因help / improve等泛词进入前5。',
  'BGE前5全部2级，分别讨论钚的用途与性质，当前标注下比其他三组前5更纯。',
  '两组重排却把谈polonium/radium并串联“science help”“technology”的0级问答片段升到第1（原BM25第60、BGE第27）。关系措辞很像，关键元素身份却不对；这是明确的实体混淆现象，不是相关材料未召回。',
  '逐字核对实体名称，并单独检查模型是否过度偏好与问题同句式的片段；不将本报告作为核技术内容的正确性评估。'),
 '2024-40868': (
  '绿色生活怎样改善自我照护',
  '需要明确green指亲近自然、环保生活还是其他意义；不能仅讲一般self-care收益。',
  'BM25第4名是低自尊危害（0级），与自我照护相邻但没有绿色生活机制；其余也有多个1级背景。',
  'BGE前5均≥1级，但大多为一般self-care介绍，只有部分绿色疗法相关；P@5高并不自动表明同时满足green限定。',
  '两组重排把绿色网站的礼篮文章排第1（1级），同时把自然疗法、居家环境等2级材料提升，前10一致。需要区分网页/品牌中的Green与正文实际说明的绿色生活机制。',
  '检查正文是否真正把自然或环保行为与自我照护联系起来；此处只评价匹配关系，不确认健康功效。'),
 '2024-213789': (
  '黑人群体器官衰竭风险的解释',
  '题目问特定人群与器官衰竭原因；泛疾病风险、肾衰竭或心衰竭的范围需要分别核对。',
  'BM25第2名“多种疾病为何更常见”在该段落上被标为0级；它匹配人群与风险，却未必回答器官衰竭的具体因果。',
  'BGE前3是3级的肾病风险段落，优于泛疾病介绍；但第5名未标注，不能把未标注直接断言为错误。',
  '两组重排前5均3级、前10一致，说明高等级材料原先已在两池内。仍有一个泛心衰材料被标为3级，应结合完整上下文与题目器官范围复核；高等级不等于医学真实性已核验。',
  '把题目范围、段落解释范围与自动等级分开记录；不要从检索标注直接推出群体医学因果结论。'),
 '2024-141577': (
  '被视为公共威胁会怎样',
  'public menace可能指公共滋扰、威胁行为或口语称呼；没有法域、对象和事实背景，无法锁定唯一法律后果。',
  'BM25前列多是menace的例句、词典定义（1级或0级），词形匹配强但不解释被认定后的后果。',
  'BGE更多转向公共秩序犯罪、威胁和扰乱秩序，仍有主体/行为范围偏差；主题接近不等于适用同一问题。',
  'BM25重排把0级vilification段落升第1，BGE重排把“危险犬/公共滋扰动物”1级段落升第1，且两组候选不同。两者对public与后果的表述产生不同偏好，实体与法域约束仍未解决。',
  '先限定人还是动物、国家地区与行为，再核对后果。网页中的罚则只作为原文证据，不在本报告中确认为适用法律建议。'),
}

def esc(value):
    return html.escape(str(value), quote=True)

def f(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'

def badge(g):
    if g is None or g < 0:
        return '<span class="badge unknown">未标注</span>'
    names = {0: '0级 · 不相关', 1: '1级 · 话题相关', 2: '2级 · 含答案信息', 3: '3级 · 直接回答'}
    return f'<span class="badge grade{g}">{names[g]}</span>'

def metric_table(metrics, keys, qid=None):
    head = ''.join(f'<th>{esc(k)}</th>' for k in keys)
    body = ''
    for method in METHODS:
        values = metrics['per_query'][method][qid] if qid else metrics['aggregate'][method]
        vals = [values[k] if qid else values[k]['mean'] for k in keys]
        body += f'<tr><th><i class="dot {method}"></i>{esc(LABELS[method])}</th>'
        for k, value in zip(keys, vals):
            others = [metrics['per_query'][m][qid][k] if qid else metrics['aggregate'][m][k]['mean'] for m in METHODS]
            valid = [v for v in others if v is not None]
            best = bool(valid) and value == (min(valid) if k.endswith('_ms') else max(valid))
            body += f'<td class="{"best" if best else ""}">{f(value, 2 if k.endswith("_ms") else 4)}</td>'
        body += '</tr>'
    return '<div class="tablewrap"><table><thead><tr><th>方案</th>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'

def main():
    rows = list(jsonl(RUN/'retrieval.jsonl'))
    docs = {d.doc_id: d for d in load_documents(RUN/'corpus.jsonl')}
    labels = load_qrels(RUN/'qrels.tsv')
    metrics, cfg = read_json(RUN/'metrics.json'), read_json(RUN/'config.json')
    lengths = read_json(ROOT/'.cache/diagnostics/fenxi-token-lengths.json')
    tokenizer = BM25Retriever(list(docs.values()), cfg['retrieval']['bm25'])
    assert set(NOTES) == {r['qid'] for r in rows}
    # Audit evidence before interpreting it.
    for r in rows:
        for base in ('bm25', 'bge'):
            assert {h['doc_id'] for h in r['methods'][base]['hits']} == {h['doc_id'] for h in r['methods'][base+'_rerank']['hits']}
    identical10 = sum([h['doc_id'] for h in r['methods']['bm25_rerank']['hits'][:10]] == [h['doc_id'] for h in r['methods']['bge_rerank']['hits'][:10]] for r in rows)
    identical5 = sum([h['doc_id'] for h in r['methods']['bm25_rerank']['hits'][:5]] == [h['doc_id'] for h in r['methods']['bge_rerank']['hits'][:5]] for r in rows)
    shared_mean = mean(len({h['doc_id'] for h in r['methods']['bm25']['hits']} & {h['doc_id'] for h in r['methods']['bge']['hits']}) for r in rows)
    nav, cases, case_stats = [], [], []
    for index, r in enumerate(rows, 1):
        qid = r['qid']; lab = labels[qid]; note = NOTES[qid]
        title, intent, lex, sem, rer, check = note
        ref = metrics['per_query']['bm25'][qid]
        available = Counter(lab.get(d) for d in docs if lab.get(d, 0) > 0)
        sets = {m: {h['doc_id'] for h in r['methods'][m]['hits']} for m in METHODS}
        common = len(sets['bm25'] & sets['bge'])
        top5 = {m: [h['doc_id'] for h in r['methods'][m]['hits'][:5]] for m in METHODS}
        top10same = [h['doc_id'] for h in r['methods']['bm25_rerank']['hits'][:10]] == [h['doc_id'] for h in r['methods']['bge_rerank']['hits'][:10]]
        rra = {h['doc_id']:h for h in r['methods']['bm25_rerank']['hits']}
        rrb = {h['doc_id']:h for h in r['methods']['bge_rerank']['hits']}
        shared_score_delta = max(abs(rra[d]['score']-rrb[d]['score']) for d in rra.keys() & rrb.keys())
        common_order_same = [h['doc_id'] for h in r['methods']['bm25_rerank']['hits'] if h['doc_id'] in rrb] == [h['doc_id'] for h in r['methods']['bge_rerank']['hits'] if h['doc_id'] in rra]
        zero_count = sum(lab.get(d) == 0 for ids in top5.values() for d in ids)
        kind = '0级前排' if zero_count else '主要检查弱相关/未标注'
        nav.append(f'<a href="#q-{qid}"><small>{index:02d}</small>{esc(title)}</a>')
        cfacts = []
        methods = []
        for method in METHODS:
            hits = r['methods'][method]['hits']; grades = [lab.get(h['doc_id']) for h in hits[:5]]
            positive = sum(g is not None and g >= 1 for g in grades)
            strong = sum(g is not None and g >= 2 for g in grades)
            zeros = sum(g == 0 for g in grades)
            unknown = sum(g is None or g < 0 for g in grades)
            first = next((h['rank'] for h in hits if lab.get(h['doc_id'], 0) >= 1), None)
            firststrong = next((h['rank'] for h in hits if lab.get(h['doc_id'], 0) >= 2), None)
            firstzero = next((h for h in hits[:10] if lab.get(h['doc_id']) == 0), None)
            info = {'method': method, 'top5_grades': grades, 'first_positive': first, 'first_strong': firststrong,
                    'positive5': positive, 'strong5': strong, 'zero5': zeros, 'unknown5': unknown}
            cfacts.append(info)
            if method.endswith('rerank'):
                inputs = lengths['reranker_pair_tokens'][qid]
                clipped = sum(inputs[h['doc_id']] > cfg['reranker']['max_length'] for h in hits[:5])
                cliptext = f'前5条中 {clipped} 条“问题＋段落”超过重排输入上限512 tokens。'
            elif method == 'bge':
                clipped = sum(lengths['bge_document_tokens'][h['doc_id']] > cfg['retrieval']['bge']['max_length'] for h in hits[:5])
                cliptext = f'前5条中 {clipped} 条段落超过BGE编码上限512 tokens。'
            else:
                cliptext = 'BM25使用标题＋完整段落的分词，不受BGE的512 tokens上限约束。'
            if firstzero:
                d = docs[firstzero['doc_id']]
                zero_note = f'前10中首个0级结果在第{firstzero["rank"]}名：{d.title or d.doc_id}。'
            else:
                zero_note = '前10没有0级结果。不要把1级或未标注条目直接写成“已判不相关”。'
            rows10, evidence = [], []
            for h in hits[:10]:
                d = docs[h['doc_id']]; g = lab.get(d.doc_id)
                movement = f'原#{h["retrieval_rank"]} → #{h["rank"]}' if method.endswith('rerank') else f'初排#{h["rank"]}'
                toks = tokenizer.tokenize([r['query'], d.search_text]); inv = {v:k for k,v in toks.vocab.items()}
                qtokens = {inv[t] for t in toks.ids[0]}; dtokens = {inv[t] for t in toks.ids[1]}
                shared = sorted(qtokens & dtokens)
                score = h['score']
                rows10.append(f'<tr><td>#{h["rank"]}</td><td>{badge(g)}</td><td>{esc(d.title or "无标题段落")}</td><td>{esc(movement)}</td><td>{score:.4f}</td></tr>')
                evidence.append(f'<details class="passage"><summary><b>#{h["rank"]}</b> {badge(g)} {esc(d.title or "无标题段落")}</summary>'
                    f'<p class="mono">{esc(d.doc_id)}</p><p class="small">{esc(movement)}；本模型原始分 {score:.4f}。与原问题共有的BM25词干：{esc(", ".join(shared) or "无")}。</p>'
                    f'<blockquote>{esc(d.search_text)}</blockquote></details>')
            text = lex if method == 'bm25' else sem if method == 'bge' else rer
            methods.append(f'<section class="method {method}"><h4><i class="dot {method}"></i>{esc(LABELS[method])}</h4>'
                f'<p class="data">前5：相关 {positive}/5 · 其中2/3级 {strong}/5 · 0级 {zeros}/5 · 未标注 {unknown}/5</p>'
                f'<p class="small">首个≥1级：{"#"+str(first) if first else "前100未找到"}；首个≥2级：{"#"+str(firststrong) if firststrong else "前100未找到"}。</p>'
                f'<p>{esc(text)}</p><p class="small"><b>核对点：</b>{esc(zero_note)}</p><p class="small">{esc(cliptext)} 输入超长只能证明发生截断，不能单凭此确认错排的原因。</p>'
                '<details class="evidence"><summary>核对前10名、分数与原文</summary><div class="tablewrap"><table><thead><tr><th>名次</th><th>标注</th><th>标题</th><th>变化</th><th>本模型分数</th></tr></thead><tbody>'
                + ''.join(rows10) + '</tbody></table></div><p class="small">下方每条可展开完整来源段落；模型分数只在同一模型内比较，负分不代表标注为0。</p>' + ''.join(evidence) + '</details></section>')
        detail = f'trec24-subset-20261027/report/q-{object_hash(qid)[:16]}.html'
        coverage = ref['corpus_relevant_coverage']
        mismatch = ('两组重排前10完全相同，说明共同候选中的强偏好已收敛；相同结果不等于两组候选池相同。' if top10same else
                    '两组重排前10并不完全相同。先看各自100候选池：一组没有的段落，重排不能补回；共同段落还应检查同分/浮点分数差异，不能断言所有差异都来自语义理解。')
        special = ''
        if qid == '2024-36355':
            b = read_json(ROOT/'.cache/diagnostics/2024-36355-bm25.json')
            d = read_json(ROOT/'.cache/diagnostics/2024-36355-rerank.json')
            special = f'<div class="experiment"><h4>已有控制变量诊断 · 不改变正式成绩</h4><div class="tablewrap"><table><tr><th>只改变什么</th><th>原结果</th><th>对照结果</th></tr><tr><td>BM25：1700\'s → 1700s</td><td>目标全库第{b["variants"]["original"]["target"]["rank"]}名</td><td>第{b["variants"]["date_only"]["target"]["rank"]}名</td></tr><tr><td>同一BGE候选池重排：sceant → scent</td><td>目标第6名</td><td>第{d["variants"]["spelling_only"]["positive_ranks"][0]}名</td></tr></table></div><p>原题重排100条顺序及分数均完整复现；仅支持输入写法会影响结果，不证明scent一定是原用户意图。</p></div>'
        flags = ' '.join(esc(x) for x in [qid, r['query'], title])
        case = f'<details class="case" id="q-{qid}" data-search="{flags}" {"open" if index == 1 else ""}><summary><span class="num">{index:02d}</span><span><strong>{esc(title)}</strong><small>{esc(qid)} · {esc(kind)}</small></span><span class="arrow">展开分析 ↓</span></summary><div class="casebody">'
        case += f'<p class="query">原问题：{esc(r["query"])}</p><p class="intent"><b>先明确要回答什么：</b>{esc(intent)}</p>'
        case += metric_table(metrics, ['precision@5','ndcg@5','ndcg@10','mrr@5','mrr@10','recall@100'], qid)
        case += f'<p class="small">绿色数值为本题该指标最高值（含并列）；它不表示答案已获事实核验。全部指标由同一份排序与标注计算。</p>'
        case += f'<div class="facts"><h4>数据能直接确认的事实</h4><p>原始标注中有 <b>{ref["relevant_total"]}</b> 条正等级段落；当前库包含 <b>{ref["relevant_in_corpus"]}</b> 条，覆盖 <b>{coverage:.2%}</b>。其中1级 {available[1]} 条、2级 {available[2]} 条、3级 {available[3]} 条。库外缺失的材料无法被任何一组找回。</p><p>BM25与BGE的100候选共有 <b>{common}</b> 条，各有 <b>{100-common}</b> 条对方没有的候选；前5共有 <b>{len(set(top5["bm25"]) & set(top5["bge"]))}</b> 条。{esc(mismatch)}</p></div>'
        case += f'<p class="small">另一个可核对细节：两次重排对共同段落的原始分数最大相差 {shared_score_delta:.6f}，共同候选的完整相对次序{"相同" if common_order_same else "有少量差异"}。当前使用FP16、分批推理；批次和数值精度可能贡献这种微小变化，未做FP32控制实验。不能把两组所有排序差异都归因于候选不同。</p>'
        case += '<h3>为什么前面会有不合适的结果？四组为什么不同？</h3><p class="small">下列判断来自原文、标注与排名。关于模型偏好/语义混淆的解释是有文本依据的推断，不是对神经网络内部机制的证明。</p><div class="methods">'+''.join(methods)+'</div>'+special
        case += f'<div class="next"><b>最值得继续核对的一步</b><p>{esc(check)}</p></div><a class="button" href="{detail}">打开原始四列报告 →</a></div></details>'
        cases.append(case)
        case_stats.append({'qid':qid,'candidate_intersection100':common,'rerank_top10_identical':top10same,'corpus_grade_counts':dict(available),'methods':cfacts})

    # An auditable summary of counts, without introducing another retrieval metric.
    countrows = ''
    for method in METHODS:
        c = Counter(labs for r in rows for h in r['methods'][method]['hits'][:5] for labs in [labels[r['qid']].get(h['doc_id'], 'unknown')])
        total = len(rows)*5
        countrows += '<tr><th>'+esc(LABELS[method])+'</th>'+''.join(f'<td>{c[g]}</td>' for g in (0,1,2,3,'unknown'))+f'<td>{sum(c[g] for g in (1,2,3))}/{total}</td></tr>'
    allkeys = ['precision@5']+[f'{m}@{k}' for m in ('ndcg','mrr','recall') for k in (5,10,20,100)]+['retrieval_ms','rerank_ms','total_ms']
    metricdata = {k:[metrics['aggregate'][m][k]['mean'] for m in METHODS] for k in allkeys}
    metric_explanations = {
      'precision': 'Precision@5只数前5中≥1级的比例，不区分1/2/3级，也不看名次先后。本轮重排更多地把正等级材料放入前5，但“5条都是1级”仍可能无法直接回答问题。',
      'ndcg': 'nDCG同时看等级与位置。同一K下，分数差来自前K的等级分布和顺序；P@5或MRR相同也可能有不同nDCG。K改变时检查范围与理想排序分母一起变化，不能只挑最有利的K。当前采用线性增益0/1/2/3。',
      'mrr': 'MRR只看第一个≥1级段落。它不区分1级背景与3级直接答案，也不考察后续材料。BGE+重排本题2024-36355的首个正等级在第6，使MRR@5为0而MRR@10为1/6；其余三组没有首个正例落在6～10名的题。',
      'recall': 'Recall分母是问题全部原始正等级标注，含库外段落。@5/10/20变化来自相关候选是否被提前；@100重排前后相同，因为重排没有改变候选集合。低覆盖率压低可达到的Recall，不能把所有损失归因于检索模型。',
      'ms': '检索耗时含BGE查询编码，但排除模型加载；重排耗时为100个候选逐批计算问题—段落分数。两组重排量级近似，差异还受文本长度、截断和运行状态影响，不是纯模型能力得分。索引构建未计入这些逐题耗时。',
    }
    header = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四种检索方案 · 数据与逐题原因分析</title><style>__CSS__</style></head><body>'''
    intro = f'''<header><div class="wrap"><p class="eyebrow">RAG / 实验解释报告 · 2026.10.28</p><h1>同一道题，为什么四种检索结果不同？</h1><p class="lead">把指标差异拆回候选、排序、原文与标注。先看整体，再逐题追查。</p><div class="hero-stats"><span><b>20</b>道问题逐题分析</span><span><b>10,000</b>个固定段落</span><span><b>4</b>种检索方案</span><span><b>800</b>个前10名证据位置</span></div></div></header>
    <div class="layout"><aside><a class="brand" href="#overview">分析导航</a><a href="#metrics">① 同一指标为何不同</a><a href="#mechanism">② 差异发生在哪一步</a><a href="#cases">③ 逐题分析 / 20题</a><div class="qnav">{''.join(nav)}</div><a href="#limits">④ 边界与下一步</a></aside><main>
    <section id="overview"><p class="eyebrow">先读这四点</p><h2>整体上重排改善了前排，但没有解决所有题意约束</h2><div class="takeaways"><article><b>前5的正等级占比提高</b><p>Precision@5：BM25 0.68 → 重排0.83；BGE 0.79 → 重排0.84。但等级1也计相关，并不保证能回答。</p></article><article><b>召回与排序是两个瓶颈</b><p>BGE的Recall@100高于BM25；重排保持各自100候选不变，只能把已有材料前移，不能找回未召回或库外材料。</p></article><article><b>模型“相似”不等于题目“满足”</b><p>地点、年代、人群、元素名称、否定立场或因果方向被忽略时，词面/语义相近的错误段落就会提前。</p></article><article><b>先检查标注与语料范围</b><p>按题平均只覆盖原始已标注相关段落的11.75%。0级、1级、未标注必须分开；自动标注也可能有实体不一致的疑点。</p></article></div>
    <div class="notice"><b>这不是完整TREC榜单或模型内部因果证明。</b>来源是公开多阶段检索候选池中的固定子集；UMBRELA是自动段落标注。本文不重新检索、不改原题、不改原排序。文本支持的解释标为推断；已有控制变量实验单独标示。</div></section>'''
    body = '<section id="metrics"><p class="eyebrow">01 / 汇总对比</p><h2>同一个指标下，四种方案为什么有不同分数？</h2>'
    body += metric_table(metrics,['ndcg@5','ndcg@10','ndcg@20','ndcg@100','precision@5','mrr@10','recall@100','total_ms'])
    body += '<p class="small">↑ 前七项越高越好；总耗时单位ms、越低越快。浅绿表示同列最高（耗时为最低），并列全部保留；20题均有效。</p>'
    options=''.join(f'<option value="{k}" {"selected" if k=="ndcg@10" else ""}>{k}</option>' for k in allkeys)
    body += f'<div class="chart"><label for="metric">换一个指标观察：<select id="metric">{options}</select></label><div id="bars">'+''.join(f'<div class="barrow"><span>{esc(LABELS[m])}</span><div class="track"><div class="bar {m}"></div></div><b class="barvalue"></b></div>' for m in METHODS)+'</div><p id="metricNote"></p></div>'
    body += '<div class="explain-grid"><article><h3>nDCG：前排等级与顺序不同</h3><p>两组重排的nDCG@10最高，但差值约0.0089，20题不足以确认稳定优势。到@100时，BGE直接检索反超BM25重排：前5集中得好，不等于100条整体更好。应同时看候选池中的相关材料数量、等级与位置。</p></article><article><h3>Precision：相关数量不同</h3><p>每组共观察100个“问题—前5段落”位置。BGE重排比BM25重排只多1个正等级位置（84与83）；平均分差0.01不能解释成模型有稳定1%的优势，更不是回答准确率。</p></article><article><h3>MRR：第一个正例的位置不同</h3><p>重排会把某些题的首个正例推后：如钚技术应用的0级实体混淆段落被推到第1。即使整体P@5更高，MRR也可能下降。1级也算正例，因此MRR接近1并不说明答案证据充分。</p></article><article><h3>Recall：能否进入100候选</h3><p>BM25与BGE的100候选平均仅共享 '+f'{shared_mean:.1f}'+' 条，因此可找回的正例不同。各自加重排的Recall@100完全一样；它们在较小K下的Recall可以改变。</p></article></div>'
    body += '<h3>前5条的标注构成：别把“话题相关”当作直接答案</h3><div class="tablewrap"><table><thead><tr><th>方案</th><th>0级</th><th>1级</th><th>2级</th><th>3级</th><th>未标注</th><th>正等级位置数</th></tr></thead><tbody>'+countrows+'</tbody></table></div><p class="small">每组20题×前5条＝100个位置，允许同一段落在不同题/方法重复出现。不是独立文档数量。1级只与话题有关、未回答；2级含答案信息；3级直接针对问题。未标注在现有指标计零增益，但不能据此断言内容无关。</p></section>'
    body += f'''<section id="mechanism"><p class="eyebrow">02 / 机制与可验证事实</p><h2>四种方案在哪一步分开，又为什么会重新接近？</h2><div class="flow"><div>同一问题<br>同一10,000段落库</div><span>→</span><div>BM25词项匹配<br>BGE向量匹配</div><span>→</span><div>各召回100条<br>候选池不同</div><span>→</span><div>保持原排序<br>或同一模型重排</div><span>→</span><div>评价前K条<br>前5条用于生成</div></div><p>BM25偏好分词后的词项重合、词频与长度；BGE将问题与段落编码成归一化向量，用FAISS精确内积检索；重排模型联合读取问题与各候选段落，重新打分。三者优化方式不同，也都不读取qrels来排序。</p><p><b>并不是每道题的四组结果都不同。</b>两组重排在 <b>{identical5}/20</b> 道题上前5顺序完全相同，在 <b>{identical10}/20</b> 道题上前10顺序完全相同。共同候选和同一重排模型可以让结果收敛；Recall@100仍可不同，因为100条完整候选集合不同。共同段落的重排分数还存在微小数值差别，各题单列最大差值；批次/FP16影响是待验证解释。</p><div class="facts"><h3>如何区分“原因已证实”和“有根据的解释”？</h3><p><span class="badge fact">事实</span>等级、原文、得分、名次变化、候选交集、分词和输入长度可直接核对。<span class="badge infer">解释</span>“偏向泛主题”“忽略了国家限定”来自问题与错排正文的对照，并非读取了模型内部思考。<span class="badge test">对照实验</span>第15题只改年份写法或疑似错词，其余条件保持一致，支持特定输入变化影响结果。</p></div><p class="small">BGE段落编码与重排输入上限均为512 tokens；逐题记录了前5有多少文本超长。超长是可核验限制，但没有额外对照时不能直接把它认定为某次错排的主因。</p></section>'''
    body += '<section id="cases"><p class="eyebrow">03 / 逐题定位</p><h2>20道题，逐条看原因与原文</h2><p>建议顺序：先读题意与指标 → 看两组候选是否不同 → 读四张方法卡 → 展开原文核对。每个问题都分析，没有0级前排的问题也会明确说明。</p><div class="toolbar"><input id="search" type="search" placeholder="搜索中文题名、英文问题或编号" aria-label="搜索问题"><button id="expandAll">展开全部问题</button><button id="collapseAll">收起全部问题</button><span id="count">20 / 20题</span></div><p id="empty" hidden>没有匹配的问题，请更换关键词。</p>'+''.join(cases)+'</section>'
    body += '''<section id="limits"><p class="eyebrow">04 / 结论的范围</p><h2>下一步应该改什么，哪些结论还不能下？</h2><ol class="steps"><li><b>先修可复现的预处理不一致。</b>第15题日期写法的影响已经实测；作为新的对照实验评估统一规范，保留本轮基线。</li><li><b>逐题核对实体与关系。</b>阅读困难/写作差、钋/钚、印度/英国、科罗拉多道路/阿尔卑斯等，都说明泛相似不能代替约束满足。若加入改写或过滤，应记录统一规则，而不是逐题手工调到高分。</li><li><b>分开检查1级和2/3级。</b>Precision与MRR使用≥1阈值，容易让话题相关背景看起来很好；本报告给出2/3级条数用于解释，但不改变正式指标阈值。</li><li><b>先补齐语料与复核疑似错标，再扩大实验。</b>对Riaño等实体题，子集缺失和自动标签疑点可能比换排序器更关键。当前结果不能直接与完整TREC榜单比较。</li><li><b>检索优劣与回答质量分开。</b>本分析聚焦检索；不把引用有效率、无截断或高nDCG当作千问回答真实正确的证明。医学、法律等网页仅作为待核对文本。</li></ol><div class="notice">没有进行模型内部归因、显著性检验、全库完整TREC评测或新的批量生成。所有逐题解释只适用于本轮20题、固定候选池子集及现用自动标注。</div><h3>原始记录与复核入口</h3><div class="links"><a href="trec24-subset-20261027/report/index.html">原始四组报告</a><a href="trec24-subset-20261027/metrics.json">全部指标</a><a href="trec24-subset-20261027/retrieval.jsonl">保存的排名与分数</a><a href="trec24-subset-20261027/dataset.json">数据来源与抽样清单</a><a href="trec24-subset-20261027/metrics_validation.json">720项指标独立核验</a></div><details><summary>报告来源哈希与统计口径</summary><pre>__PROVENANCE__</pre></details></section>'''
    provenance = {'run':'trec24-subset-20261027','analysis_date':'2026-10-28 Asia/Shanghai','documents':len(docs),'queries':len(rows),'method_query_combinations':len(rows)*4,'displayed_evidence_positions':len(rows)*4*10,'same_reranked_top10_queries':identical10,'average_bm25_bge_common_candidates100':shared_mean,'sha256':{name:digest(RUN/name) for name in ['retrieval.jsonl','metrics.json','qrels.tsv','corpus.jsonl','config.json']},'no_retrieval_or_generation_rerun':True}
    body=body.replace('__PROVENANCE__',esc(json.dumps(provenance,ensure_ascii=False,indent=2)))
    data=json.dumps({'metrics':metricdata,'notes':metric_explanations,'labels':[LABELS[m] for m in METHODS]},ensure_ascii=False).replace('<','\\u003c')
    footer='<footer>fenxi.html · 数据、判断与边界分开展示 · 离线可读，无外部脚本依赖</footer></main></div>'
    document=header.replace('__CSS__',CSS)+intro+body+footer+'<script>const DATA='+data+';'+JS+'</script></body></html>'
    atomic_text(OUT,document)
    atomic_text(ROOT/'.cache/diagnostics/fenxi-audit.json',json.dumps({'provenance':provenance,'cases':case_stats},ensure_ascii=False,indent=2)+'\n')
    print(f'Created {OUT}; {len(cases)} complete cases; {OUT.stat().st_size:,} bytes.')

CSS = '''
:root{--bg:#f5f6fa;--ink:#162436;--muted:#637185;--line:#dfe5ee;--bm25:#2563eb;--bge:#008980;--bm25_rerank:#b77911;--bge_rerank:#7852c6}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:20px}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}header{background:linear-gradient(120deg,#102039,#193b54);color:white;padding:52px 28px 40px}header .wrap{max-width:1440px;margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700;color:#65809a;margin:0 0 8px}header .eyebrow{color:#a9d8db}h1{font-size:clamp(26px,3.2vw,42px);line-height:1.35;margin:12px 0}.lead{color:#c8d8e4;font-size:17px}.hero-stats{display:flex;gap:34px;flex-wrap:wrap;margin-top:25px}.hero-stats b{font-size:27px;margin-right:9px;color:#93e2d2}.layout{display:grid;grid-template-columns:225px minmax(0,1fr);max-width:1500px;margin:auto;padding:32px 24px;gap:28px}aside{position:sticky;top:18px;align-self:start;max-height:94vh;overflow:auto;border-right:1px solid var(--line);padding-right:16px;font-size:13px}aside a{display:block;padding:6px 8px;color:#52657c;border-radius:6px}aside a:hover{background:#e7eef8}.brand{font-size:17px;font-weight:750;color:var(--ink)}.qnav{border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:10px 0;padding:8px 0}.qnav small{color:#8a97a6;margin-right:9px;font-variant-numeric:tabular-nums}main{min-width:0}section[id]{scroll-margin-top:24px}main>section{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin-bottom:24px;box-shadow:0 3px 12px #12223404}h2{font-size:25px;line-height:1.45;margin:6px 0 18px}h3{font-size:18px;margin:24px 0 12px}h4{font-size:15px;margin:0 0 10px}p{margin:10px 0}a{color:#265aba;text-decoration:none}a:hover{text-decoration:underline}.small,small{font-size:12px;color:var(--muted);line-height:1.7}.takeaways,.explain-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.takeaways article,.explain-grid article{border:1px solid var(--line);border-radius:10px;padding:16px 18px;background:#fbfcfe}.takeaways article b{font-size:16px}.takeaways p,.explain-grid p{font-size:14px;margin:6px 0 0}.notice{background:#fff7e8;border-left:4px solid #dda449;border-radius:8px;padding:15px 18px;margin:20px 0;font-size:13px}.facts{background:#edf5fb;border:1px solid #d5e7f5;padding:17px 20px;border-radius:10px;margin:18px 0;font-size:14px}.facts h3{margin-top:0}.facts h4{color:#245475}.tablewrap{overflow:auto;max-width:100%;border:1px solid var(--line);border-radius:9px;margin:15px 0}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th,td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:#f1f5f9;font-size:12px;white-space:nowrap}td{font-variant-numeric:tabular-nums}tbody tr:last-child>*{border-bottom:0}th:first-child{min-width:140px}.best{background:#e8f5ed;color:#106347;font-weight:700}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;background:#666}.dot.bm25,.bar.bm25{background:var(--bm25)}.dot.bge,.bar.bge{background:var(--bge)}.dot.bm25_rerank,.bar.bm25_rerank{background:var(--bm25_rerank)}.dot.bge_rerank,.bar.bge_rerank{background:var(--bge_rerank)}.chart{padding:20px;background:#f8fafc;border-radius:10px;margin:22px 0}select,input,button{font:inherit;font-size:14px;border:1px solid #c8d3df;border-radius:7px;background:white;color:var(--ink);padding:9px 12px}select{margin-left:12px}button{cursor:pointer}button:hover{background:#edf3fb}.barrow{display:grid;grid-template-columns:150px 1fr 78px;gap:12px;align-items:center;margin:15px 0;font-size:13px}.track{height:13px;background:#e4eaf2;border-radius:5px;overflow:hidden}.bar{height:100%;border-radius:5px;transition:width .2s}.barvalue{text-align:right}#metricNote{font-size:13px}.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:16px;background:#f6f8fc;border-radius:10px}.flow div{padding:10px 14px;background:white;border:1px solid var(--line);border-radius:8px;font-size:13px;text-align:center}.flow span{color:#7d91aa}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:22px 0}.toolbar input{flex:1;min-width:230px}#count{font-size:12px;color:var(--muted)}.case{border:1px solid var(--line);border-radius:11px;margin:14px 0;overflow:hidden;background:white}.case>summary{display:flex;gap:14px;align-items:center;padding:18px;cursor:pointer;list-style:none;background:#fafbfd}.case>summary::-webkit-details-marker{display:none}.case>summary:hover{background:#f0f5fc}.case>summary strong{font-size:16px}.case>summary small{display:block}.num{font-size:20px;color:#708499;font-variant-numeric:tabular-nums}.arrow{margin-left:auto;white-space:nowrap;font-size:12px;color:#5c7391}.case[open] .arrow{color:#2563eb}.casebody{padding:8px 20px 24px;border-top:1px solid var(--line)}.query{font-size:14px;background:#f1f4f8;padding:12px 15px;border-radius:7px;overflow-wrap:anywhere}.intent{font-size:14px}.methods{display:grid;grid-template-columns:1fr 1fr;gap:14px}.method{border:1px solid var(--line);border-radius:10px;padding:17px;min-width:0;border-top:3px solid #9ba8b8}.method.bm25{border-top-color:var(--bm25)}.method.bge{border-top-color:var(--bge)}.method.bm25_rerank{border-top-color:var(--bm25_rerank)}.method.bge_rerank{border-top-color:var(--bge_rerank)}.method>p{font-size:14px}.method>.small{font-size:12px}.data{background:#f5f7fa;border-radius:6px;padding:9px;font-size:12px!important}.badge{display:inline-block;border-radius:5px;padding:1px 6px;font-size:11px;line-height:1.8;white-space:nowrap;margin:2px 3px 2px 0}.grade0{background:#fff0ee;color:#ac3e35}.grade1{background:#edf3fe;color:#3c6098}.grade2{background:#e8f5ee;color:#23764d}.grade3{background:#d1eedd;color:#155c3a}.unknown{background:#eef0f3;color:#626c79}.fact{background:#dcedfa;color:#245475}.infer{background:#fff0d7;color:#9a611a}.test{background:#e9e0ff;color:#6d46a8}details.evidence>summary,details.passage>summary{cursor:pointer;color:#285c98;font-size:13px;padding:8px 0}.evidence table{font-size:11px}.evidence th:first-child{min-width:35px}.evidence th,.evidence td{padding:6px}.passage{border-top:1px solid var(--line)}.passage summary{overflow-wrap:anywhere}.mono{font:11px/1.6 Consolas,monospace;overflow-wrap:anywhere;color:#6b788a}blockquote{margin:10px 0;padding:12px 14px;background:#f7f8fa;border-left:3px solid #a5b8cc;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:380px;overflow:auto}.next{background:#f0f8f3;border:1px solid #d4e8db;border-radius:8px;padding:15px 18px;font-size:14px;margin:20px 0}.button{display:inline-block;border:1px solid #cbd8ea;padding:8px 14px;border-radius:7px;font-size:13px}.experiment{background:#f5f0ff;border:1px solid #ddd0f5;padding:18px;border-radius:10px;margin:18px 0}.steps{padding-left:22px}.steps li{padding:7px 0}.links{display:flex;gap:15px;flex-wrap:wrap;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:18px;font-size:12px}footer{padding:12px 0 35px;font-size:12px;color:#8492a4}details[hidden]{display:none!important}@media(max-width:1100px){.layout{grid-template-columns:1fr}aside{display:none}}@media(max-width:740px){header{padding:30px 18px}.layout{padding:18px 10px}main>section{padding:19px 15px}.takeaways,.explain-grid,.methods{grid-template-columns:1fr}h2{font-size:22px}.casebody{padding:8px 12px 18px}.barrow{grid-template-columns:120px 1fr 65px;font-size:12px}.hero-stats{gap:14px}.hero-stats span{font-size:12px}.hero-stats b{font-size:22px}.arrow{display:none}.case>summary{padding:14px;gap:10px}}@media print{aside,.toolbar,.chart select{display:none}.layout{display:block}details.case{break-before:auto}main>section{box-shadow:none}.methods{display:block}.method{margin:10px 0}header{background:#18334c!important}body{background:white}}
'''

JS = '''
const metric=document.getElementById('metric');
function draw(){const key=metric.value;const vals=DATA.metrics[key];const maximum=Math.max(...vals.filter(v=>v!==null),0.000001);document.querySelectorAll('.barrow').forEach((row,i)=>{row.querySelector('.bar').style.width=(vals[i]===null?0:vals[i]/maximum*100)+'%';row.querySelector('.barvalue').textContent=vals[i]===null?'N/A':vals[i].toFixed(key.endsWith('_ms')?2:4)});const kind=key.endsWith('_ms')?'ms':key.split('@')[0];document.getElementById('metricNote').textContent=DATA.notes[kind]+' 图中条长按当前四组最大值缩放。';}
metric.addEventListener('change',draw);draw();
const cases=[...document.querySelectorAll('details.case')];
const search=document.getElementById('search');
function filterCases(){const query=search.value.trim().toLowerCase();let n=0;cases.forEach(c=>{c.hidden=!c.dataset.search.toLowerCase().includes(query);if(!c.hidden)n++});document.getElementById('count').textContent=n+' / '+cases.length+'题';document.getElementById('empty').hidden=n>0}
search.addEventListener('input',filterCases);
document.getElementById('expandAll').addEventListener('click',()=>cases.filter(c=>!c.hidden).forEach(c=>c.open=true));
document.getElementById('collapseAll').addEventListener('click',()=>cases.forEach(c=>c.open=false));
function reveal(){const id=decodeURIComponent(location.hash.slice(1));const target=document.getElementById(id);if(target&&target.classList.contains('case')){if(target.hidden){search.value='';filterCases()}target.open=true;target.scrollIntoView({block:'start'})}}
window.addEventListener('hashchange',reveal);reveal();
'''

if __name__ == '__main__':
    main()
