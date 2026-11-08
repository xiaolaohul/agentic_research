"""Single entry point. Run `python main.py --help` for all stages."""
import argparse
import importlib.util
import json
import platform
import sys
import time

from src.utils import load_config, path, storage_check, versions


def parser():
    p = argparse.ArgumentParser(description='TREC RAG 2024：四种检索方案与离线 HTML 报告')
    p.add_argument('--config', default='configs/default.yaml', help='配置文件，相对路径以项目目录为基准')
    commands = p.add_subparsers(dest='command')
    commands.add_parser('doctor', help='检查检索环境、GPU 和磁盘')
    download = commands.add_parser('download', help='下载评测文件、公开候选正文或检索模型')
    download.add_argument('asset', choices=['metadata', 'pool', 'bge', 'reranker'])
    prep = commands.add_parser('prepare', help='从段落 JSONL 或公开候选池建立固定子集')
    prep.add_argument('--source', help='本地 JSONL / JSONL.GZ 文件')
    prep.add_argument('--pool', action='store_true', help='source 使用 {query,candidates} 格式')
    prep.add_argument('--pool-scope', choices=['selected', 'all'], help='语料来自选中问题的候选，或来自全部来源问题的候选并集')
    prep.add_argument('--topics', help='覆盖默认的官方 topics 文件路径')
    prep.add_argument('--qrels', help='覆盖默认的 segment 级 qrels 文件路径')
    prep.add_argument('--sample', action='store_true', help='创建人工样例；不当作 TREC 数据')
    prep.add_argument('--download', action='store_true', help='先下载官方 topics 和 segment qrels')
    prep.add_argument('--overwrite', action='store_true', help='明确替换已有处理后数据')
    index = commands.add_parser('index', help='创建或复用 BM25/BGE 索引')
    index.add_argument('--method', choices=['all', 'bm25', 'bge'], default='all')
    index.add_argument('--force', action='store_true')
    retrieval = commands.add_parser('retrieve', help='运行四组检索、评估并导出报告（需要已有索引）')
    retrieval.add_argument('--run-id')
    hybrid = commands.add_parser('hybrid', help='从已完成的四组实验补充RRF，另存五组报告，不运行模型')
    hybrid.add_argument('--source-run', required=True, help='来源实验编号或目录')
    hybrid.add_argument('--run-id', required=True, help='新实验编号，不覆盖已有实验')
    run = commands.add_parser('run', help='建索引 → 四组检索 → 评估 → HTML')
    run.add_argument('--run-id')
    report = commands.add_parser('report', help='读取已有结果，重新导出 HTML，不重新推理')
    report.add_argument('--run')
    report.add_argument('--verify-metrics', action='store_true', help='使用 sklearn 独立核对逐问题 nDCG、Recall 和 Precision@5')
    return p


def doctor(cfg):
    diagnostics = {'python': sys.executable, 'python_version': platform.python_version(),
                   'storage': storage_check(cfg), 'packages': versions(), 'warnings': []}
    from src.telemetry import memory_snapshot
    diagnostics['system_memory'] = memory_snapshot()
    try:
        import torch
        diagnostics['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diagnostics['gpu'] = torch.cuda.get_device_name(0)
            diagnostics['vram_gib'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        else:
            diagnostics['warnings'].append('当前 PyTorch 没有可用 CUDA；需将 bge/reranker.device 改为 cpu 或安装 CUDA 版 PyTorch。')
    except ImportError:
        diagnostics['warnings'].append('尚未安装 PyTorch。')
    for module in ('yaml', 'bm25s', 'faiss', 'sentence_transformers', 'transformers'):
        if not importlib.util.find_spec(module):
            diagnostics['warnings'].append(f'缺少模块：{module}')
    diagnostics['prepared_dataset'] = path(cfg['dataset']['manifest']).is_file()
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def download_model(cfg, asset):
    from huggingface_hub import HfApi, snapshot_download
    import fnmatch
    model_cfg = cfg['retrieval']['bge'] if asset == 'bge' else cfg['reranker']
    info = HfApi().model_info(model_cfg['model'], revision=model_cfg['revision'], files_metadata=True)
    patterns = ['*.json', '*.safetensors', '*.txt', '*.model', 'README.md']
    total = sum(f.size or 0 for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in patterns)
                and not f.rfilename.startswith(('onnx/', 'openvino/')))
    storage_check(cfg, total)
    result = retry_download(lambda: snapshot_download(model_cfg['model'], revision=info.sha, allow_patterns=patterns,
                               ignore_patterns=['onnx/*', 'openvino/*'], max_workers=2))
    print(f'{asset}: {result}\nResolved revision: {info.sha}')


def retry_download(action, attempts=3):
    from requests.exceptions import RequestException
    for attempt in range(attempts):
        try:
            return action()
        except RequestException as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401, 403, 404) or attempt == attempts - 1:
                raise ValueError('模型下载失败；缓存中的未完成片段已保留。网络恢复后重新运行同一 download 命令可续传。'
                                 f' 原因：{type(exc).__name__}, HTTP {status or "N/A"}') from exc
            print(f'下载连接中断，正在继续已有缓存（重试 {attempt + 1}/{attempts - 1}）...', flush=True)
            time.sleep(2 ** attempt)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command is None:
        from src.menu import menu
        return menu(args.config)
    cfg = load_config(args.config)
    if args.command == 'doctor':
        doctor(cfg)
    elif args.command == 'download':
        if args.asset == 'metadata':
            from src.dataset import download_assets
            download_assets(cfg)
        elif args.asset == 'pool':
            from src.sources import download_pool
            download_pool(cfg)
        else:
            download_model(cfg, args.asset)
    elif args.command == 'prepare':
        from src.dataset import download_assets, prepare, prepare_sample
        if args.pool_scope and not args.pool:
            raise ValueError('--pool-scope 需要与 --pool 一起使用。')
        if args.sample:
            if args.source or args.pool or args.download or args.topics or args.qrels:
                raise ValueError('--sample 不能与真实数据参数一起使用。')
            prepare_sample(cfg, args.overwrite)
        else:
            if args.download:
                download_assets(cfg)
            if not args.source:
                raise ValueError('需要 --source 指向本地段落/候选池文件；--download 只下载评测文件。')
            prepare(cfg, path(args.source), args.topics, args.qrels, args.pool, args.overwrite, args.pool_scope)
    elif args.command == 'hybrid':
        from src.pipeline import resolve_run
        from src.hybrid import supplement_rrf
        supplement_rrf(resolve_run(cfg, args.source_run), cfg, args.run_id)
    elif args.command == 'index':
        from src.pipeline import build_indexes
        build_indexes(cfg, ('bm25', 'bge') if args.method == 'all' else (args.method,), args.force)
    elif args.command in ('run', 'retrieve'):
        from src.pipeline import build_indexes, retrieve
        from src.report import generate_report
        if args.command == 'run':
            build_indexes(cfg)
        run = retrieve(cfg, args.run_id)
        generate_report(run)
    else:
        from src.pipeline import resolve_run
        from src.report import generate_report
        run = resolve_run(cfg, args.run)
        generate_report(run, verify_metrics=args.verify_metrics)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已中断，已保存的检索结果保留在实验目录。', file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        sys.exit(2)
