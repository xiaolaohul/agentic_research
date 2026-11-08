# RAG_1：TREC RAG 2024 四组检索对比

这个项目比较 **BM25、BGE、BM25＋重排序、BGE＋重排序**。所有操作通过 `main.py` 完成，报告是能离线打开的 HTML。

## 100题 RRF 混合检索基线

五组实验在原四组上补充 **BM25＋BGE（RRF）**。两路各取Top-100，按段落ID去重，等权累加 `1/(60+原名次)`，同分按段落ID排序，最终保留Top-100。缺少某一路时该路贡献为0；不相加或归一化两种模型的原始分数，不读取相关性标签参与融合。常数60固定，未在这100题上调参。公式依据：[Cormack等，SIGIR 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)。

从已完成的100题实验补充RRF，不重新运行模型：

```powershell
.\.venv\Scripts\python.exe main.py --config configs/trec24-100q-rrf-20261105.yaml hybrid --source-run trec24-100q-20261029 --run-id trec24-100q-rrf-20261105
```

新目录必须不存在；重复实验请使用另一个编号。源实验保持原样，新目录保存独立的语料、问题、标注、配置及五组结果。`artifact_audit.json`记录来源文件哈希和原四组指标一致性，`metrics_validation.json`记录独立指标核验。结果入口为 `results/trec24-100q-rrf-20261105/report/index.html`。重新导出使用：

```powershell
.\.venv\Scripts\python.exe main.py report --run trec24-100q-rrf-20261105 --verify-metrics
```

`retrieval.rrf`明确记录 `enabled: true`、`rank_constant: 60`、`input_k: 100`、`output_k: 100`。启用此配置执行普通`run`命令时，也会从本次两路检索结果计算RRF。旧四组实验继续按各自存档中的方案评测。

`reranker.depth: 100`表示BM25和BGE分别对各自前100条进行模型重排；每题另存`reranked_count`，少于100条时如实记录。它与配对输入上限512 tokens及评测nDCG@5/@10不同。融合本身不是神经模型重排。本轮包含5组，没有RRF＋模型重排。

RRF每条结果记录两路原名次、原始分数、贡献和融合分数。`fusion_ms`为本次CPU融合实测耗时；复用结果的`retrieval_ms`为历史BM25与BGE检索耗时之和，`total_ms`再加本次融合耗时，属于串行组合估算，不是本次端到端实测，也不是并行耗时。原两路计时和模型版本保留来源信息。

新报告沿用当前100题实验的子集nDCG口径、Judged@10和逐题比较图，隐藏分批汇总；后台仍计算全部已有指标。只评测检索，不调用千问。

目标机器：Windows、RTX 3060 Laptop 6GB、系统内存 16GB。默认一次处理一个问题，检索和重排序分阶段使用显卡。默认使用固定语料子集。

## 先运行什么

**本机日常使用：双击项目根目录的 `启动RAG.bat`，输入数字选择操作即可。** 它自动使用项目虚拟环境，不需要激活或记忆参数。

菜单支持查看已有报告、检查环境和新建检索实验。查看报告时可选择已有实验。新建检索实验自动分配编号。

在已经激活项目环境的 VS Code 终端，也可以只运行 `python main.py` 打开同一个菜单。也可以用命令行直接运行检索和导出报告。

在项目目录打开 PowerShell。下文命令中的 `python` 指项目环境的 Python：

```powershell
# 本机已创建 .venv，复用 E:\Conda_envs\mytorch 中现有的 CUDA PyTorch。
.\.venv\Scripts\Activate.ps1
python main.py doctor
```

早期的本地问答实验记录见 [PHASE2_RESULTS.md](PHASE2_RESULTS.md)。

第三阶段已接入真实 TREC RAG 2024 候选段落子集，使用默认配置；手写样例仍由 `configs/sample.yaml` 独立管理。第三阶段实验编号为 `trec24-subset-20261027`，结果与限制见 [PHASE3_RESULTS.md](PHASE3_RESULTS.md)。

如果 PowerShell 的脚本策略不允许激活，不必更改系统策略，直接使用：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

**第一步可以先跑人工样例，熟悉 HTML 界面。** 样例使用真实 BM25/BGE/重排序算法，但样例文本和相关性标签是手工编写的，不是 TREC 成绩。

```powershell
python main.py --config configs/sample.yaml prepare --sample
python main.py --config configs/sample.yaml run
```

第一次运行 BGE 和 Reranker 需要下载模型。结果目录打印到终端；打开其中的 `report/index.html`。样例与真实数据使用不同目录和索引，不会互相覆盖。

## 安装到另一台电脑

使用 Python 3.10 或 3.11 建立环境。先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)安装与你的 NVIDIA 驱动兼容的 CUDA PyTorch，然后：

```powershell
python -m pip install -r requirements.txt
python main.py doctor
```

`requirements-lock.txt` 记录本次验证环境中新增依赖的精确版本；复现时可安装该文件。基础 NumPy 为 2.2.6、PyTorch 为 2.5.1+cu121，它们来自已有 Conda 环境，未包含在本地增量锁定文件中。

本机 `.venv` 使用 `--system-site-packages` 复用已有的 torch 2.5.1+cu121，其他依赖安装在 `.venv` 内。不要把这个虚拟环境直接复制到另一台电脑。

## 真实 TREC RAG 2024 数据

来源：

- [TREC 2024 任务说明](https://trec-rag.github.io/trec24/)
- [NIST 2024 数据目录](https://trec.nist.gov/data/rag2024.html)
- [2024 UMBRELA 段落标注发布页](https://trec-rag.github.io/annoucements/umbrela-qrels/)
- [MS MARCO v2.1 语料说明及官方入口](https://trec-rag.github.io/annoucements/2025-rag25-corpus/)：该说明确认沿用 2024 的语料。

TREC RAG 2024 有 301 个测试问题。项目默认使用 **MS MARCO v2.1 segment 粒度**，因此接入 **2024 UMBRELA segment qrels**。它们是模型评判标注，不能描述成人工真值。NIST 同页还发布 document retrieval qrels，不可直接把其父文档 ID 当作 segment ID 使用。

下载小型官方评测文件：

```powershell
python main.py download metadata
```

此命令只下载 topics、qrels，并写入来源链接与 SHA256；**不包含段落正文**。不自动下载 25GB 的完整语料压缩包或 84GB 的 BM25 索引。

项目已经接入发布方 [Castorini Ragnarök 的测试候选正文](https://github.com/castorini/ragnarok_data/tree/7432266dadad8368b6f35d9d4d0f3c0e2b69297c/rag24/retrieve_results/RANK_ZEPHYR)：固定 Git 提交与 LFS SHA256，约 58MB，覆盖 301 个问题、30,100 个候选记录、30,013 个不同段落。这是其他模型筛选过的 Top-100 候选池，不是完整 MS MARCO 语料。

在新环境首次准备时，可运行：

```powershell
python main.py download metadata
python main.py download pool
python main.py prepare --source data/raw/rag24-test-candidate-pool.jsonl --pool
```

**本机已完成准备，无需重复 prepare。** 默认从整个发布候选池抽取 10,000 个不同段落，并从可用官方问题中固定抽取 20 个；不使用 qrels 分数筛选段落或问题。子集来源、版本、校验值、抽样范围和官方 ID 都保存在 manifest。其他模型对候选池的预筛选会产生选择偏差，四组结论只适用于这个固定子集。

准备真实段落正文有两种输入方式：

1. 从官方 MS MARCO v2.1 segmented corpus 取得 JSONL / JSONL.GZ 文件或已有子集。
2. 从可追溯的公开检索候选缓存中提取段落。候选池存在来源偏差，应在报告中明确，不作为完整语料上的公平基准结论。

原始官方 segment 行可直接读取：

```json
{"docid":"msmarco_v2.1_doc_00_100#0_200","title":"Title","url":"https://example.org","segment":"Original segment text"}
```

也支持统一格式：

```json
{"doc_id":"msmarco_v2.1_doc_00_100#0_200","text":"Original segment text","title":"Title","url":"https://example.org","parent_id":"msmarco_v2.1_doc_00_100"}
```

以上 ID 仅演示格式。不要自行构造 ID 代替官方 ID。正文必须是真实段落，不从测试答案生成语料。

```powershell
python main.py prepare --source data/raw/your_segments.jsonl --download
python main.py run
```

公开候选池格式为：

```json
{"query":{"qid":"2024-...","text":"Question"},"candidates":[{"docid":"msmarco_v2.1_doc_...#...","doc":{"segment":"Text","title":"Title","url":"https://example.org"}}]}
```

```powershell
python main.py prepare --source data/raw/your_candidate_pool.jsonl --pool --download
```

默认按固定种子从可用的官方 topics 选最多 20 个问题，去重并固定抽取最多 10,000 个段落。默认 `dataset.pool_scope: all` 从全部来源问题的候选并集建库，使语料选择独立于本次评测问题；`--pool-scope selected` 可明确改为只使用选中问题的候选。四组方法在同一份语料中重新检索，不直接沿用来源方法的排名。抽样不读取相关性标签，候选池的问题 ID 和文字必须与官方 topics 对齐。

`data/subsets/manifest.json` 记录具体问题、段落 ID、来源文件哈希、抽样方法和限制。若 qrels 与段落 ID 完全不匹配，会拒绝准备数据。如果存在部分匹配，报告展示每个问题的相关段落覆盖率；低覆盖率时不得据此声称某个模型检索能力差。

修改规模、路径或模型请编辑 `configs/default.yaml`。重新准备数据需要明确使用 `--overwrite`。已有实验快照不受影响。

## 四组实验与运行入口

```powershell
python main.py index                         # 两类索引
python main.py index --method bm25           # 只建 BM25
python main.py retrieve                      # 四组检索 + 评估 + 报告
python main.py run                           # 建索引 + 四组检索 + 评估 + 报告
python main.py report --run YOUR_RUN_ID      # 只重新导出 HTML
python main.py report --run YOUR_RUN_ID --verify-metrics  # 用 sklearn 独立核对 nDCG/Recall/Precision@5
```

查看当前真实子集的已有结果：

```powershell
python main.py report --run trec24-subset-20261027 --verify-metrics
```

BM25 使用 bm25s 的 Lucene BM25、英文停用词、英文词干化，默认 k1=0.9、b=0.4；不是对 Anserini 分词与得分的逐位复现。BGE 使用 bge-base-en-v1.5、固定查询指令、归一化向量和 FAISS CPU 精确内积检索。两者对同一份 `title + segment` 文本建索引，BGE 按模型上限截断，并记录配置。

候选数默认 100；两组基础结果各缓存一次，再交给同一个 bge-reranker-v2-m3 重排全部候选。保留初始排名与初始分数。原始分数不可在不同模型间直接比较。BGE 查询编码计入检索耗时；模型加载排除在逐问题耗时之外。

BGE 的建库模型 commit 会记录在索引中，并用于后续查询编码，防止 `main` 变化导致向量空间不一致。配置与数据指纹不一致时拒绝加载旧索引。正式实验建议将配置中的模型 `revision` 改为固定 commit。

## 结果目录与 HTML

每次实验写入 `results/<实验编号>/`：

```text
run.json                    运行状态、软件版本、模型 revision
config.json                 实验配置快照
dataset.json                固定子集清单与来源
corpus.jsonl / queries.jsonl / qrels.tsv    数据快照
bm25.candidates.jsonl        第一阶段 BM25 候选
bge.candidates.jsonl         第一阶段 BGE 候选
retrieval.jsonl              四组完整排序、初始排名和分数
metrics.json / metrics.csv  汇总与逐问题指标
metrics_validation.json     sklearn 对 nDCG/Recall/Precision@5 的独立数值核验
paired_analysis.json        逐问题上升/下降/持平及可跳转案例
*.trec                      可交给外部评估器的标准 run 格式
report/index.html           总览页
report/q-*.html             每个问题的四列并排详情
```

HTML 无外部脚本或 CDN，直接双击打开。支持问题搜索、指标切换、排名变化、相同段落突出显示和原文展开。原始段落按普通文本转义，避免来源文本被当成 HTML 执行。已有问答实验的报告仍可读取已保存的答案和证据。

100题报告增加“重排退步案例与原因分类 · nDCG@10”：分别覆盖BM25的18题和BGE的32题，提供主类统计、对应问题ID、逐题原文解释、排名变化与逐段计分贡献。主分类不重复计数，次要标签允许重叠；未标注与0级分别处理，语义解释不等于已证明模型内部因果。内容快照为实验目录内的`rerank_drop10.json`，解释源文件为`tools/rerank_drop_notes.py`。重新导出报告会检查分析与排序、标签和配置的指纹，避免展示过期解释；若输入改变，先运行`python tools/analyze_rerank_drops.py --extract --build`。该命令只检查已存排名和本地分词，不重新运行检索或生成模型。

检索指标：线性增益 nDCG@5/10/20/100、MRR@K、Recall@K、Precision@5，并展示有效问题数、标注率和语料覆盖率。

- **当前100题报告与新实验默认使用子集内nDCG**（`evaluation.ndcg_scope: corpus`）：理想排序仅使用同一固定语料内、针对当前问题的正等级标注，四种方法共用基准；不使用各自检索出的100条作为理想排序的范围。子集内没有正等级标注时记为N/A。这些分数不直接与完整语料评测比较。
- 历史配置未指定`ndcg_scope`时保持原始口径`all_judgments`，避免无提示改变旧实验。报告和指标JSON明确记录采用的口径。完整标注仍保留，未改变数据抽样或标签。
- 当前100题报告展示nDCG、MRR、Judged@10，不展示Recall、Precision；Judged@10＝前10条中针对当前问题有标注记录的段落数÷10，等级0也计入。其他历史报告中的Recall仍使用当前问题全部已标注相关段落作分母。
- Precision@5＝前5条中等级≥1的段落数÷5；未标注和不足5条的空位计不相关。没有已标注相关正例的问题记为 N/A，与其他主指标的汇总口径一致。
- 重新导出报告会按实验保存的排序、标注和配置重算指标，同步更新 JSON/CSV/HTML；不启动检索或生成模型。
- 未标注在指标中计零增益，页面显示“未标注”；负等级表示未判定，会跳过。
- 没有相关正例的指标为 N/A，不把缺标注算作零分，汇总报告各指标的有效问题数。
- TREC 导出用倒数名次作为排序分，保证外部评估器按相同顺序处理原始分数并列；原始分数保存在 JSONL。
- 引用 ID 有效率只检查引用是否来自实际输入，不等于正确性、忠实性或官方 nugget 得分。

第一版未实现官方 nugget 自动评估。需要匹配的官方 nuggets、评估工具及额外评判模型后再扩展，不以生成模型自评分代替正式评估。

## 验证与限制

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 graded nDCG 的已知结果、完整 qrels 分母、缺标注、重复 ID、segment 映射、候选池去重、数据指纹以及 BM25 保存加载。集成测试临时创建微型未训练 BERT 编码器和重排序器，实际执行 BM25、向量编码、FAISS、重排序及报告导出。测试使用本地创建的模型权重，检验检索和报告流程。

若连接中断，下载命令会重试并保留缓存；BGE 和 Reranker 通过 Hugging Face 缓存续传，TREC 候选正文通过分块校验和下载清单续传。网络恢复后再次执行相同命令可以续传。不要把未完成权重当作可用模型。

默认磁盘工作预算 40GiB，保留 20GiB 空余。检查不是操作系统级硬配额；长时间模型自动下载仍需留意磁盘。模型缓存统一放在项目 `models/huggingface`，不重复下载不同精度版本。

**子集结果不是完整 TREC 成绩。** 样例结果不是 TREC 数据。完整语料和完整向量索引应迁移到服务器或使用更多存储，不能在当前 80GB 空间内承诺完整基准复现。
