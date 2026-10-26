"""Download the original Physical Intelligence base checkpoint with MD5 checks."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

import requests

destination = Path(os.environ["ROBOTWIN_WORKSPACE"]) / "models/robotwin_pi0/pi0_base_jax"
destination.mkdir(parents=True, exist_ok=True)
prefix = "checkpoints/pi0_base/"
source_path = destination / "download-source.json"
if source_path.exists():
    items = json.loads(source_path.read_text())["objects"]
else:
    session = requests.Session()
    session.trust_env = False  # Direct GCS access was measured faster on this host.
    response = session.get("https://storage.googleapis.com/storage/v1/b/openpi-assets/o",
                           params={"prefix": prefix, "maxResults": 1000}, timeout=60)
    response.raise_for_status()
    listing = response.json()
    assert not listing.get("nextPageToken"), "Incomplete checkpoint listing"
    items = listing["items"]
    source_path.write_text(json.dumps({
        "source": "gs://openpi-assets/checkpoints/pi0_base", "objects": items,
    }, indent=2) + "\n")


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def fetch(item):
    session = requests.Session()
    session.trust_env = False
    path = destination / item["name"][len(prefix):]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    expected_md5 = item["md5Hash"]
    if path.exists():
        assert path.stat().st_size == expected_size and md5(path) == expected_md5, path
        return
    partial = path.with_name(path.name + ".partial")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            assert offset <= expected_size, partial
            if offset < expected_size:
                url = "https://storage.googleapis.com/openpi-assets/" + quote(item["name"], safe="/")
                with session.get(url, params={"generation": item["generation"]},
                                  headers={"Range": f"bytes={offset}-"} if offset else {},
                                  stream=True, timeout=(30, 120)) as result:
                    result.raise_for_status()
                    if offset and result.status_code != 206:
                        raise RuntimeError("Server did not honor download resume range")
                    with partial.open("ab" if offset else "wb") as stream:
                        for block in result.iter_content(1024 * 1024):
                            stream.write(block)
            assert partial.stat().st_size == expected_size, partial
            assert md5(partial) == expected_md5, f"MD5 mismatch: {partial}"
            partial.rename(path)
            print(f"verified {path.relative_to(destination)} ({expected_size} bytes)", flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, items))
print(f"Base checkpoint verified: {destination}", flush=True)
