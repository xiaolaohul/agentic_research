"""Public TREC RAG 2024 candidate text, pinned to its publisher's Git/LFS version."""
from urllib.parse import quote

from .deployment import ranged_download
from .utils import path, write_json

POOL_REPOSITORY = 'castorini/ragnarok_data'
POOL_REVISION = '7432266dadad8368b6f35d9d4d0f3c0e2b69297c'
POOL_FILE = ('rag24/retrieve_results/RANK_ZEPHYR/'
             'retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl')
POOL_SHA256 = '7f675eadda6e99fa7990c2034734e87d7380c3ac6ef9d51da0c11bdae84253f9'
POOL_BYTES = 57876985


def download_pool(cfg):
    url = f'https://media.githubusercontent.com/media/{POOL_REPOSITORY}/{POOL_REVISION}/{quote(POOL_FILE, safe="/")}'
    target = path(cfg['dataset']['raw_dir']) / 'rag24-test-candidate-pool.jsonl'
    ranged_download(url, target, POOL_SHA256, cfg, size=POOL_BYTES)
    write_json(target.with_suffix('.source.json'), {
        'repository': POOL_REPOSITORY, 'revision': POOL_REVISION, 'path': POOL_FILE,
        'url': url, 'sha256': POOL_SHA256, 'bytes': POOL_BYTES,
        'method': 'fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf',
        'scope': 'published multi-stage top-100 candidate pool, TREC RAG 2024 test',
        'limitation': 'The publisher selected these candidates. This is not the full MS MARCO collection.'})
    print(f'Candidate text pool: {target}', flush=True)
    return target
