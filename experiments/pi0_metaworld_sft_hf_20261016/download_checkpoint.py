"""Download the reference SFT checkpoint from the publisher's Hugging Face SFT repository."""
import concurrent.futures
import hashlib
import fcntl
import json
import os
from pathlib import Path
import time
import threading

import requests

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "checkpoints/RLinf-Pi0-MetaWorld-SFT"
REVISION = "d0b8fbe41877a8105e2fd91e750e44542e6855bd"
BASE = f"https://hf-mirror.com/RLinf/RLinf-Pi0-MetaWorld-SFT/resolve/{REVISION}"
SIZE = 7011543960
SHA256 = "c606538b48414b00af1966dd8cf27fe0a87893f390619cac6bf0920a7cf06e2d"
CHUNK = 64 * 1024 * 1024
TRANSFER_CONDITION = threading.Condition()
ACTIVE_TRANSFERS = 0


def scheduled_part(index):
    global ACTIVE_TRANSFERS
    with TRANSFER_CONDITION:
        while ACTIVE_TRANSFERS >= 12:
            TRANSFER_CONDITION.wait(timeout=1)
        ACTIVE_TRANSFERS += 1
    try:
        download_part(index)
    finally:
        with TRANSFER_CONDITION:
            ACTIVE_TRANSFERS -= 1
            TRANSFER_CONDITION.notify_all()


def download_part(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    part = DEST / "parts" / f"{index:04d}"
    for attempt in range(8):
        offset = part.stat().st_size if part.exists() else 0
        if offset == end - start + 1:
            return
        try:
            # Include the byte offset in the URL to prevent a proxy reusing another range.
            url = f"{BASE}/model.safetensors?part={index}&offset={offset}"
            with requests.Session() as http:
                http.trust_env = False
                response = http.get(url, headers={"Range": f"bytes={start + offset}-{end}"},
                                    stream=True, timeout=(20, 60))
                try:
                    response.raise_for_status()
                    expected = f"bytes {start + offset}-{end}/{SIZE}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Invalid range response: {response.status_code}, {response.headers.get('Content-Range')}")
                    with part.open("ab") as file:
                        for block in response.iter_content(1024 * 1024):
                            file.write(block)
                finally:
                    response.close()
            if part.stat().st_size != end - start + 1:
                raise RuntimeError("Truncated range")
            print(json.dumps({"event": "part_complete", "part": index,
                              "downloaded": sum(p.stat().st_size for p in part.parent.iterdir())}), flush=True)
            return
        except Exception as error:
            print(json.dumps({"event": "download_retry", "part": index, "attempt": attempt + 1,
                              "error": str(error)[:300]}), flush=True)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Part {index} failed after 8 attempts")


def main():
    lock = (ROOT / "checkpoint_download.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"event": "download_already_running"}), flush=True)
        return
    (ROOT / "download.pid").write_text(str(os.getpid()))
    (DEST / "parts").mkdir(parents=True, exist_ok=True)
    (DEST / "lerobot/metaworld_mt50").mkdir(parents=True, exist_ok=True)
    for name in ["lerobot/metaworld_mt50/norm_stats.json"]:
        if (DEST / name).exists():
            expected_norm_hash = "ab3e2f7380ccbd0967fe391e07d4bc67492ceb7e3e06a9c03397aa6c28309220"
            if hashlib.sha256((DEST / name).read_bytes()).hexdigest() == expected_norm_hash:
                continue
        with requests.Session() as http:
            http.trust_env = False
            response = http.get(f"{BASE}/{name}", timeout=40)
        response.raise_for_status()
        (DEST / name).write_bytes(response.content)
    model = DEST / "model.safetensors"
    if not model.exists():
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            list(pool.map(scheduled_part, range((SIZE + CHUNK - 1) // CHUNK)))
        with model.with_suffix(".partial").open("wb") as output:
            for part in sorted((DEST / "parts").iterdir()):
                with part.open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
        os.replace(model.with_suffix(".partial"), model)
    digest = hashlib.file_digest(model.open("rb"), "sha256").hexdigest()
    if model.stat().st_size != SIZE or digest != SHA256:
        raise RuntimeError(f"Checkpoint verification failed: {digest}")
    manifest = {"repository": "RLinf/RLinf-Pi0-MetaWorld-SFT", "hub": "Hugging Face",
                "revision": REVISION, "sha256": digest, "bytes": SIZE,
                "role": "official published SFT checkpoint, frozen evaluation",
                "source": BASE, "local_training": False}
    (DEST / "verified.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "checkpoint_verified", **manifest}), flush=True)


if __name__ == "__main__":
    main()
