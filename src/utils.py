import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def read_json(file):
    return json.loads(Path(file).read_text(encoding='utf-8-sig'))


def write_json(file, value):
    atomic_text(file, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def atomic_text(file, text):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    # Windows readers / virus scanners can briefly deny replacement of an open file.
    # Unique temp names also prevent competing writers from sharing a staging file.
    temp = file.with_name(file.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(text, encoding='utf-8')
        for attempt in range(10):
            try:
                temp.replace(file)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(min(0.05 * 2**attempt, 0.5))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except PermissionError:
            pass


def jsonl(file):
    with Path(file).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{file}:{number}: invalid JSON') from exc


def write_jsonl(file, rows):
    atomic_text(file, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def digest(file):
    result = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def setup_environment(cfg):
    cache = path(cfg['storage']['model_cache'])
    os.environ.setdefault('HF_HOME', str(cache))
    os.environ.setdefault('HF_ENDPOINT', cfg['storage'].get('hf_endpoint', 'https://huggingface.co'))
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def versions():
    output = {}
    for package in ('numpy', 'torch', 'bm25s', 'faiss-cpu', 'sentence-transformers', 'transformers'):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = None
    return output


def storage_check(cfg, additional_bytes=0):
    free = shutil.disk_usage(ROOT).free
    reserve = cfg['storage']['reserve_gib'] * 1024**3
    used = sum(f.stat().st_size for name in ('data', 'indexes', 'models', 'tools', 'results', '.venv')
               for f in (ROOT / name).rglob('*') if f.is_file())
    if free - additional_bytes < reserve:
        raise ValueError('磁盘空间不足：将侵占配置中的保留空间。')
    if used + additional_bytes > cfg['storage']['budget_gib'] * 1024**3:
        raise ValueError('项目文件超过配置的磁盘预算。')
    return {'free_gib': round(free / 1024**3, 2), 'project_gib': round(used / 1024**3, 2)}


def load_config(file):
    import yaml
    cfg = yaml.safe_load(path(file).read_text(encoding='utf-8-sig'))
    for key in ('candidate_k', 'context_k', 'display_k'):
        if not isinstance(cfg['retrieval'][key], int) or cfg['retrieval'][key] < 1:
            raise ValueError(f'retrieval.{key} must be a positive integer')
    if cfg['retrieval']['context_k'] > cfg['retrieval']['candidate_k']:
        raise ValueError('context_k cannot exceed candidate_k')
    if max(cfg['evaluation']['ks']) > cfg['retrieval']['candidate_k']:
        raise ValueError('评估 K 不能超过 candidate_k，否则无法评价截断后的排序。')
    depth = cfg['reranker'].get('depth', cfg['retrieval']['candidate_k'])
    if type(depth) is not int or not 1 <= depth <= cfg['retrieval']['candidate_k']:
        raise ValueError('reranker.depth 必须在1到candidate_k之间。')
    rrf = cfg['retrieval'].get('rrf', {})
    if rrf.get('enabled', False):
        if any(type(rrf.get(key)) is not int or rrf[key] < 1 for key in ('rank_constant', 'input_k', 'output_k')):
            raise ValueError('RRF参数必须为正整数。')
        if rrf['input_k'] > cfg['retrieval']['candidate_k'] or max(cfg['evaluation']['ks']) > rrf['output_k']:
            raise ValueError('RRF输入深度不能超过候选数，输出数量不能小于评测K。')
    setup_environment(cfg)
    return cfg
