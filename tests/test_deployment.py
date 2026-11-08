from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import unittest

from src.deployment import ranged_download
from src.utils import read_json, write_json
from tests.test_core import temporary_directory


@contextmanager
def range_server(payload, wrong_range=False):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            value = self.headers.get('Range', '')
            match = re.fullmatch(r'bytes=(\d+)-(\d+)', value)
            calls.append(value)
            if not match:
                self.send_error(400)
                return
            lower, upper = map(int, match.groups())
            body = payload[lower:upper + 1]
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {0 if wrong_range else lower}-{upper}/{len(payload)}')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/artifact', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class DeploymentTests(unittest.TestCase):
    cfg = {'storage': {'reserve_gib': 0, 'budget_gib': 100000}}
    payload = bytes(range(256)) * 200

    def test_parallel_download_full_hash_and_verified_cache(self):
        checksum = hashlib.sha256(self.payload).hexdigest()
        with temporary_directory() as folder, range_server(self.payload) as (url, calls):
            dest = Path(folder) / 'artifact.bin'
            ranged_download(url, dest, checksum, self.cfg, chunk_bytes=8192, workers=3)
            self.assertEqual(dest.read_bytes(), self.payload)
            count = len(calls)
            ranged_download(url, dest, checksum, self.cfg, chunk_bytes=8192, workers=3)
            self.assertEqual(len(calls), count)
            self.assertTrue(read_json(dest.with_name(dest.name + '.download.json'))['verified'])

    def test_resume_skips_committed_ranges(self):
        checksum = hashlib.sha256(self.payload).hexdigest()
        with temporary_directory() as folder, range_server(self.payload) as (url, calls):
            dest = Path(folder) / 'resume.bin'
            with dest.with_name(dest.name + '.part').open('wb') as out:
                out.write(self.payload[:8192])
                out.truncate(len(self.payload))
            write_json(dest.with_name(dest.name + '.download.json'), {
                'identity': {'url': url, 'bytes': len(self.payload), 'sha256': checksum, 'chunk_bytes': 8192},
                'completed': [0]})
            ranged_download(url, dest, checksum, self.cfg, size=len(self.payload), chunk_bytes=8192, workers=2)
            self.assertNotIn('bytes=0-8191', calls)
            self.assertEqual(dest.read_bytes(), self.payload)

    def test_wrong_ranges_cannot_publish_a_model(self):
        checksum = hashlib.sha256(self.payload).hexdigest()
        with temporary_directory() as folder, range_server(self.payload, wrong_range=True) as (url, calls):
            dest = Path(folder) / 'wrong.bin'
            with self.assertRaisesRegex(ValueError, 'Range'):
                ranged_download(url, dest, checksum, self.cfg, size=len(self.payload), chunk_bytes=8192, workers=2)
            self.assertFalse(dest.exists())

    def test_wrong_sha_cannot_publish_a_model(self):
        with temporary_directory() as folder, range_server(self.payload) as (url, calls):
            dest = Path(folder) / 'wrong-hash.bin'
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                ranged_download(url, dest, '0' * 64, self.cfg, size=len(self.payload), chunk_bytes=8192, workers=2)
            self.assertFalse(dest.exists())


if __name__ == '__main__':
    unittest.main()
