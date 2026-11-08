"""Resumable HTTP range downloads with SHA256 verification."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import threading
import time

from .utils import digest, read_json, storage_check, utc_now, write_json

class RangeMismatch(ValueError):
    """A server returning different bytes cannot be treated as a transient timeout."""


def _range_session(url):
    import requests
    from urllib.parse import urlparse
    session = requests.Session()
    if urlparse(url).hostname in ('127.0.0.1', 'localhost', '::1'):
        session.trust_env = False
    session.headers.update({'User-Agent': 'RAG-1-research/2.0', 'Accept-Encoding': 'identity'})
    return session


def ranged_download(url, destination, expected_sha256, cfg, size=None, workers=8, chunk_bytes=8 * 1024**2):
    """Validate every Content-Range and the final hash before publishing a file.

    Each worker writes disjoint byte ranges; the journal commits only complete writes.
    No executable or model is used until its complete SHA256 matches the publisher.
    """
    destination = Path(destination)
    if destination.is_file():
        if digest(destination) != expected_sha256:
            raise ValueError(f'已有文件校验失败，拒绝覆盖：{destination}')
        print(f'Verified cache: {destination.name}', flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        with _range_session(url) as session:
            with session.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
                if response.status_code != 206 or not match:
                    raise ValueError('下载源没有返回可靠的 HTTP Range；不能安全地并行续传。')
                size = int(match[1])
    if not size or chunk_bytes < 1 or workers < 1:
        raise ValueError('下载大小、分块大小和并发数必须为正。')
    part = destination.with_name(destination.name + '.part')
    journal_file = destination.with_name(destination.name + '.download.json')
    identity = {'url': url, 'bytes': size, 'sha256': expected_sha256, 'chunk_bytes': chunk_bytes}
    if part.exists() or journal_file.exists():
        if not part.is_file() or not journal_file.is_file():
            raise ValueError(f'续传文件不完整，请检查 {part} 和 {journal_file}')
        journal = read_json(journal_file)
        if journal.get('identity') != identity or part.stat().st_size != size:
            raise ValueError('续传文件与源版本/校验信息不一致，拒绝混合不同版本。')
    else:
        storage_check(cfg, size)
        # Pre-allocation makes disk budgeting independent of download order.
        with part.open('wb') as stream:
            stream.truncate(size)
        journal = {'identity': identity, 'completed': [], 'started_at': utc_now()}
        write_json(journal_file, journal)
    total_chunks = (size + chunk_bytes - 1) // chunk_bytes
    completed = set(journal['completed'])
    if any(not isinstance(i, int) or i < 0 or i >= total_chunks for i in completed):
        raise ValueError('下载续传清单包含无效块编号。')
    lock, stopped = threading.Lock(), threading.Event()
    started, initial_bytes, last_progress = time.monotonic(), sum(min(chunk_bytes, size - i * chunk_bytes) for i in completed), [0.0]
    print(f'Download {destination.name}: {size / 1024**3:.2f} GiB, {len(completed)}/{total_chunks} chunks cached', flush=True)

    def fetch(index):
        import requests
        lower, upper = index * chunk_bytes, min(size, (index + 1) * chunk_bytes) - 1
        last_error = None
        for attempt in range(6):
            if stopped.is_set():
                return
            try:
                with _range_session(url) as session:
                    with session.get(url, headers={'Range': f'bytes={lower}-{upper}'}, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        wanted = f'bytes {lower}-{upper}/{size}'
                        if response.status_code != 206 or response.headers.get('Content-Range') != wanted:
                            raise RangeMismatch(f'Range 校验失败：期望 {wanted}，收到 {response.headers.get("Content-Range")}')
                        data = bytearray()
                        for block in response.iter_content(256 * 1024):
                            if stopped.is_set():
                                return
                            data.extend(block)
                            if len(data) > upper - lower + 1:
                                raise ValueError('下载分块超过预期长度。')
                        if len(data) != upper - lower + 1:
                            raise ValueError('下载分块长度不足。')
                with part.open('r+b', buffering=0) as stream:
                    stream.seek(lower)
                    written = stream.write(data)
                    if written != len(data):
                        raise OSError('下载分块未完整写入磁盘。')
                    os.fsync(stream.fileno())
                with lock:
                    completed.add(index)
                    journal['completed'] = sorted(completed)
                    write_json(journal_file, journal)
                    now = time.monotonic()
                    if now - last_progress[0] > 5 or len(completed) == total_chunks:
                        done = sum(min(chunk_bytes, size - i * chunk_bytes) for i in completed)
                        rate = (done - initial_bytes) / max(0.001, now - started) / 1024**2
                        print(f'{destination.name}: {done / size:.1%} ({rate:.1f} MiB/s)', flush=True)
                        last_progress[0] = now
                return
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if isinstance(exc, RangeMismatch):
                    break
                if isinstance(exc, requests.HTTPError) and exc.response.status_code in (401, 403, 404):
                    break
                if attempt < 5:
                    stopped.wait(min(2**attempt, 10))
        raise ValueError(f'分块 {index} 下载失败，已完成内容可续传：{last_error}') from last_error

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(fetch, i) for i in range(total_chunks) if i not in completed]
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stopped.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True)
    print(f'Verify SHA256: {destination.name}', flush=True)
    if digest(part) != expected_sha256:
        raise ValueError(f'完整文件 SHA256 校验失败，未启用：{part}')
    part.replace(destination)
    journal.update(completed_at=utc_now(), verified=True)
    write_json(journal_file, journal)
    return destination
