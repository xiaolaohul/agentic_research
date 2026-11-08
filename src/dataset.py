"""TREC imports preserve segment IDs; relevance labels never select the corpus."""
from dataclasses import asdict
import gzip
import heapq
import json
from pathlib import Path
import re
import urllib.request

from .schemas import Document, Query
from .utils import digest, jsonl, object_hash, path, storage_check, utc_now, write_json, write_jsonl, atomic_text


def normalize_document(row):
    content = row.get('doc', row)
    doc_id = str(row.get('doc_id', row.get('docid', row.get('id', ''))))
    text = content.get('segment', content.get('text', content.get('contents', '')))
    if not doc_id or not isinstance(text, str) or not text.strip():
        raise ValueError('段落必须有 doc_id/docid/id 和 text/segment/contents；不把整篇 body 冒充 segment。')
    return Document(doc_id, text.strip(), str(content.get('title', '')),
                    str(content.get('url', '')), str(row.get('parent_id', doc_id.split('#')[0])))


def load_documents(file):
    docs = [normalize_document(row) for row in jsonl(file)]
    ids = [d.doc_id for d in docs]
    if not docs or len(ids) != len(set(ids)):
        raise ValueError('语料为空或存在重复段落 ID。')
    return docs


def load_queries(file):
    p = Path(file)
    if p.suffix == '.jsonl':
        queries = [Query(str(r.get('qid', r.get('id', ''))), str(r.get('text', r.get('query', ''))))
                   for r in jsonl(p)]
    else:
        queries = []
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                qid, text = line.split('\t', 1)
                queries.append(Query(qid, text.strip()))
    if not queries or any(not q.qid or not q.text for q in queries) or len({q.qid for q in queries}) != len(queries):
        raise ValueError('问题数据为空、字段缺失或存在重复 qid。')
    return queries


def load_qrels(file):
    result = {}
    if not Path(file).exists():
        return result
    for line in Path(file).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError('qrels 必须是四列：qid iteration segment_id grade。')
        qid, _, doc_id, grade = fields
        grade = int(grade)
        if grade < 0:  # Negative grades mean unjudged, not explicit nonrelevance.
            continue
        previous = result.setdefault(qid, {}).get(doc_id)
        if previous is not None and previous != grade:
            raise ValueError(f'冲突的 qrels：{qid}, {doc_id}')
        result[qid][doc_id] = grade
    return result


def download(url, destination, cfg, max_bytes=32 * 1024**2):
    destination = Path(destination)
    if destination.exists():
        return destination
    storage_check(cfg, max_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + '.part')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'RAG-1-research/1.0'})
        with urllib.request.urlopen(request, timeout=60) as source, temp.open('wb') as out:
            if int(source.headers.get('Content-Length', '0')) > max_bytes:
                raise ValueError(f'下载超过预算上限：{url}')
            count = 0
            while block := source.read(1024**2):
                count += len(block)
                if count > max_bytes:
                    raise ValueError(f'下载超过预算上限：{url}')
                out.write(block)
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return destination


def download_assets(cfg):
    d = cfg['dataset']
    raw = path(d['raw_dir'])
    for key, filename in [('topics_url', 'topics.rag24.test.txt'), ('qrels_url', 'qrels.rag24.umbrela.txt')]:
        print(f'Download {filename}', flush=True)
        download(d[key], raw / filename, cfg)
    topics = load_queries(raw / 'topics.rag24.test.txt')
    labels = load_qrels(raw / 'qrels.rag24.umbrela.txt')
    if not all(q.qid.startswith('2024-') for q in topics):
        raise ValueError('下载的 topics 不是 TREC RAG 2024。')
    if not any('#' in doc for grades in labels.values() for doc in grades):
        raise ValueError('下载的 qrels 不是 segment 粒度，拒绝与 segment 检索结果混用。')
    write_json(raw / 'sources.json', {
        'downloaded_at': utc_now(), 'topics_url': d['topics_url'], 'qrels_url': d['qrels_url'],
        'judgments': d['judgments'], 'topics_sha256': digest(raw / 'topics.rag24.test.txt'),
        'qrels_sha256': digest(raw / 'qrels.rag24.umbrela.txt'),
        'topics': len(topics), 'judged_queries': len(labels)})


def iter_source(file):
    opener = gzip.open if str(file).endswith('.gz') else open
    with opener(file, 'rt', encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{file}:{number}: invalid JSON') from exc


def save_prepared(cfg, docs, queries, labels, provenance, overwrite=False):
    d = cfg['dataset']
    targets = [path(d[k]) for k in ('corpus', 'queries', 'qrels', 'manifest')]
    if not overwrite and any(p.exists() for p in targets):
        raise ValueError('处理后数据已存在。若要替换，请明确使用 --overwrite。旧索引会因指纹变化而失效。')
    ids = {doc.doc_id for doc in docs}
    if not docs or not queries:
        raise ValueError('没有可用段落或问题。')
    if len(ids) != len(docs):
        raise ValueError('段落 ID 重复。')
    qids = {q.qid for q in queries}
    labels = {q: grades for q, grades in labels.items() if q in qids}
    if labels and not any(ids.intersection(grades) for grades in labels.values()):
        raise ValueError('qrels 与子集的段落 ID 完全不匹配，拒绝输出误导性评估。')
    if provenance['kind'] != 'synthetic':
        if any(not q.qid.startswith('2024-') for q in queries):
            raise ValueError('此配置要求 TREC RAG 2024 的问题 ID。')
        if any(not re.match(r'^msmarco_v2\.1_doc_.+#.+', doc.doc_id) for doc in docs):
            raise ValueError('真实语料必须保留 MS MARCO v2.1 官方 segment ID（含 #）。')
    storage_check(cfg)
    write_jsonl(targets[0], (asdict(doc) for doc in sorted(docs, key=lambda x: x.doc_id)))
    write_jsonl(targets[1], (asdict(q) for q in queries))
    # Keep ALL original judgments for selected queries, including outside the subset.
    atomic_text(targets[2], ''.join(f'{q}\t0\t{doc}\t{grade}\n' for q in sorted(labels)
                for doc, grade in sorted(labels[q].items())))
    manifest = {
        'created_at': utc_now(), 'dataset': d['name'], 'scope': 'restricted-corpus experiment',
        'judgments': 'synthetic fixture labels' if provenance['kind'] == 'synthetic' else d['judgments'],
        'documents': len(docs), 'queries': len(queries), 'seed': d['seed'],
        'corpus_sha256': digest(targets[0]), 'queries_sha256': digest(targets[1]),
        'qrels_sha256': digest(targets[2]), 'document_ids': sorted(ids),
        'query_ids': [q.qid for q in queries], 'provenance': provenance,
        'warning': '人工样例，仅验证程序，不代表 TREC 成绩。' if provenance['kind'] == 'synthetic'
        else '固定子集实验；受子集选择与标注覆盖影响，不能当作完整 TREC 基准成绩。'}
    write_json(targets[3], manifest)
    print(f'Prepared {len(docs)} segments, {len(queries)} queries.', flush=True)
    return manifest


def prepare(cfg, source, topics=None, qrels=None, pool=False, overwrite=False, pool_scope=None):
    d = cfg['dataset']
    topic_file = path(topics) if topics else path(d['raw_dir']) / 'topics.rag24.test.txt'
    qrel_file = path(qrels) if qrels else path(d['raw_dir']) / 'qrels.rag24.umbrela.txt'
    queries = load_queries(topic_file)
    official_queries = {q.qid: q for q in queries}
    available_pool_qids = set()
    scope = pool_scope or d.get('pool_scope', 'selected')
    if scope not in ('selected', 'all'):
        raise ValueError('pool_scope must be selected or all')
    if pool:
        # Availability is checked before deterministic query selection. Qrels are not read here.
        for row in iter_source(source):
            query = row.get('query', {})
            qid = str(query.get('qid', row.get('qid', '')))
            if qid not in official_queries:
                continue
            text = query.get('text')
            if text and re.sub(r'\s+', ' ', text).strip().casefold() != re.sub(r'\s+', ' ', official_queries[qid].text).strip().casefold():
                raise ValueError(f'候选池问题文字与官方 topics 不一致：{qid}')
            available_pool_qids.add(qid)
        queries = [q for q in queries if q.qid in available_pool_qids]
    # Stable hash selection is independent of labels and input order.
    queries.sort(key=lambda q: object_hash([d['seed'], q.qid]))
    queries = queries[:d['max_queries']]
    selected_qids = {q.qid for q in queries}
    heap, pool_qids, count, seen_pool = [], set(), 0, {}
    for row in iter_source(source):
        candidates = [row]
        if pool:
            qid = str(row.get('query', {}).get('qid', row.get('qid', '')))
            if qid not in official_queries or (scope == 'selected' and qid not in selected_qids):
                continue
            pool_qids.add(qid)
            candidates = row.get('candidates', [])
        for candidate in candidates:
            doc = normalize_document(candidate)
            if pool:
                signature = object_hash(asdict(doc))
                if doc.doc_id in seen_pool:
                    if seen_pool[doc.doc_id] != signature:
                        raise ValueError(f'候选池中同一 ID 对应不同文本：{doc.doc_id}')
                    continue
                seen_pool[doc.doc_id] = signature
            count += 1
            priority = int(object_hash([d['seed'], doc.doc_id]), 16)
            entry = (-priority, count, doc)
            if len(heap) < d['max_documents']:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
    if pool:
        queries = [q for q in queries if q.qid in pool_qids]
    docs = {}
    for _, _, doc in heap:
        if doc.doc_id in docs and docs[doc.doc_id] != doc:
            raise ValueError(f'同一段落 ID 对应不同文本：{doc.doc_id}')
        docs[doc.doc_id] = doc
    source_metadata = Path(source).with_suffix('.source.json')
    from .utils import read_json
    origin = read_json(source_metadata) if source_metadata.is_file() else None
    source_hash = digest(source)
    if origin and origin.get('sha256') != source_hash:
        raise ValueError('候选池正文与来源记录的 SHA256 不一致。')
    return save_prepared(cfg, list(docs.values()), queries, load_qrels(qrel_file), {
        'kind': 'candidate_pool' if pool else 'corpus_sample', 'source': str(Path(source).resolve()),
        'source_sha256': source_hash, 'publisher': origin, 'topics_sha256': digest(topic_file),
        'qrels_source': str(qrel_file), 'qrels_source_sha256': digest(qrel_file) if qrel_file.exists() else None,
        'sampling': 'seeded SHA256 priority sampling; no relevance labels used',
        'pool_scope': scope if pool else None, 'pool_available_queries': len(available_pool_qids) if pool else None,
        'corpus_query_scope': 'all source topics, independently of selected evaluation queries' if pool and scope == 'all' else 'selected evaluation queries',
        'source_bias': '候选池来源限制了可检索段落；不同方法共用其并集。' if pool else '源文件范围限制了检索空间。',
        'rows_scanned': count}, overwrite)


def validate_prepared(cfg):
    d = cfg['dataset']
    from .utils import read_json
    manifest = read_json(path(d['manifest']))
    for key, field in [('corpus', 'corpus_sha256'), ('queries', 'queries_sha256'), ('qrels', 'qrels_sha256')]:
        if digest(path(d[key])) != manifest[field]:
            raise ValueError(f'{key} 已被修改，与子集 manifest 不一致，请重新 prepare。')
    docs, queries, labels = load_documents(path(d['corpus'])), load_queries(path(d['queries'])), load_qrels(path(d['qrels']))
    return docs, queries, labels, manifest


def prepare_sample(cfg, overwrite=False):
    """Transparent synthetic fixture; uses real algorithms but never claims benchmark validity."""
    entries = [
        ('solar', 'Solar energy', 'Solar panels convert sunlight into electricity using photovoltaic cells. Batteries can store surplus electricity for use at night.'),
        ('wind', 'Wind energy', 'Wind turbines convert the kinetic energy of moving air into electrical energy. Wind is a renewable energy source.'),
        ('hydro', 'Hydropower', 'Hydropower generates electricity by passing flowing water through turbines. Dams can store water for later use.'),
        ('battery', 'Energy storage', 'A battery stores chemical energy and converts it into electrical energy. Rechargeable batteries help store intermittent solar power.'),
        ('photosynthesis', 'Photosynthesis', 'Plants use sunlight, water and carbon dioxide to produce sugars and oxygen through photosynthesis in chloroplasts.'),
        ('chlorophyll', 'Plant pigments', 'Chlorophyll is a green pigment in chloroplasts that absorbs light energy for photosynthesis.'),
        ('roots', 'Plant roots', 'Roots anchor plants and absorb water and minerals from soil. Water travels upward through xylem.'),
        ('bm25', 'Lexical retrieval', 'BM25 ranks documents using query term frequency, inverse document frequency and document length normalization. It matches lexical terms.'),
        ('dense', 'Dense retrieval', 'Dense retrieval embeds queries and documents into vectors. Similarity search can retrieve passages with related meaning even when wording differs.'),
        ('rerank', 'Cross encoder reranking', 'A cross encoder jointly reads a query and each candidate passage to score relevance. It reorders retrieved candidates but cannot recover missing passages.'),
        ('faiss', 'Vector indexing', 'FAISS is a library for similarity search over dense vectors. An exact inner product index can rank normalized vectors by cosine similarity.'),
        ('quant', 'Model quantization', 'Quantization represents neural network weights with fewer bits, reducing storage and memory requirements while potentially reducing accuracy.'),
        ('mars', 'Mars', 'Mars is the fourth planet from the Sun. Iron oxide on its surface gives the planet a red appearance.'),
        ('moon', 'Moon', 'The Moon is the natural satellite of Earth. Its gravity contributes to ocean tides.'),
        ('rain', 'Water cycle', 'Evaporation turns liquid water into vapor. Condensation forms clouds and precipitation returns water to the ground.'),
        ('cpu', 'Computing hardware', 'A CPU executes program instructions. A GPU runs many operations in parallel and can accelerate neural network inference.'),
    ]
    docs = [Document('sample-' + key, text, title) for key, title, text in entries]
    queries = [Query('sample-1', 'How do solar panels generate electricity and store it for night?'),
               Query('sample-2', 'How do plants use sunlight to make food?'),
               Query('sample-3', 'How does semantic vector retrieval differ from BM25 keyword matching?'),
               Query('sample-4', 'Can reranking retrieve a relevant passage missing from the candidate set?')]
    labels = {'sample-1': {'sample-solar': 3, 'sample-battery': 2, 'sample-wind': 0},
              'sample-2': {'sample-photosynthesis': 3, 'sample-chlorophyll': 2, 'sample-roots': 1},
              'sample-3': {'sample-bm25': 3, 'sample-dense': 3, 'sample-faiss': 1},
              'sample-4': {'sample-rerank': 3, 'sample-dense': 0, 'sample-mars': 0}}
    return save_prepared(cfg, docs, queries, labels,
                         {'kind': 'synthetic', 'description': 'Handwritten English smoke-test fixture, not TREC data'}, overwrite)
