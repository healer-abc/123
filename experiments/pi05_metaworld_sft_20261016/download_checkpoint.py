"""Resumable, range-verified parallel download of RLinf π0.5 MetaWorld SFT."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT=Path(__file__).resolve().parent
REVISION='60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e'
REPOSITORY='RLinf/RLinf-Pi05-MetaWorld-SFT'
BASE=f'https://hf-mirror.com/{REPOSITORY}/resolve/{REVISION}'
DEST=ROOT/'checkpoints/RLinf-Pi05-MetaWorld-SFT'
SIZE=7473091464
SHA256='6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221'
CHUNK=64*1024*1024
WORKERS=12


def get(url, headers=None):
    # Direct mirror route was measured at 2.46 MB/s for one 8 MiB range;
    # direct HF timed out, proxied HF ~0.60 MB/s. Verify every byte range.
    session=requests.Session();session.trust_env=False
    try:
        response=session.get(url,headers=headers,stream=True,timeout=(20,90))
        response.raise_for_status()
        return session,response
    except Exception:
        session.close();raise


def part(index):
    start=index*CHUNK;end=min(SIZE,start+CHUNK)-1
    target=DEST/'parts'/f'{index:04d}'
    length=end-start+1
    for attempt in range(12):
        have=target.stat().st_size if target.exists() else 0
        if have==length:return
        if have>length:raise ValueError(f'Part {index} oversized')
        try:
            url=f'{BASE}/model.safetensors?part={index}&offset={have}'
            session,response=get(url,{'Range':f'bytes={start+have}-{end}'})
            try:
                expected=f'bytes {start+have}-{end}/{SIZE}'
                if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                    raise ValueError(f'Bad range response {response.status_code} {response.headers.get("Content-Range")}')
                with target.open('ab') as output:
                    for block in response.iter_content(1024*1024):
                        output.write(block)
            finally:
                response.close();session.close()
            if target.stat().st_size!=length:raise IOError('Short part')
            print(json.dumps({'part':index,'complete':length}),flush=True)
            return
        except Exception as error:
            print(json.dumps({'part':index,'retry':attempt+1,'error':str(error)[:180]}),flush=True)
            time.sleep(min(2**attempt,20))
    raise RuntimeError(f'Part {index} failed after retries')


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'parts').mkdir(exist_ok=True)
    lock=(ROOT/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('metadata.pt','lerobot/metaworld_mt50/norm_stats.json'):
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size:continue
        session,response=get(f'{BASE}/{name}')
        try:target.write_bytes(response.content)
        finally:response.close();session.close()
    metadata_expected='3ef33cebb9c4c7bd21e134d9f54c255187cea4072ec74824898e014763aae617'
    assert sha256(DEST/'metadata.pt')==metadata_expected
    model=DEST/'model.safetensors'
    if not model.exists():
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(part,range((SIZE+CHUNK-1)//CHUNK)))
        assembled=model.with_suffix('.partial')
        with assembled.open('wb') as output:
            for index in range((SIZE+CHUNK-1)//CHUNK):
                with (DEST/'parts'/f'{index:04d}').open('rb') as source:
                    for block in iter(lambda:source.read(8*1024*1024),b''):output.write(block)
        os.replace(assembled,model)
    digest=sha256(model)
    if model.stat().st_size!=SIZE or digest!=SHA256:raise ValueError('Weight SHA256 or size mismatch')
    manifest={'repository':REPOSITORY,'revision':REVISION,'source':BASE,'bytes':SIZE,
              'sha256':digest,'metadata_sha256':metadata_expected,
              'norm_stats_sha256':sha256(DEST/'lerobot/metaworld_mt50/norm_stats.json'),
              'local_training':False}
    (DEST/'verified.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'event':'checkpoint_verified',**manifest}),flush=True)


if __name__=='__main__':main()
