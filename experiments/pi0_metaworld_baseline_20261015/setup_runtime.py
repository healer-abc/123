"""Finish runtime setup and restart resumable model transfer outside the chat process."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/zhsu/jyluo/projects/openpi/.venv/bin/python"


def restart_download():
    pid_file = ROOT / "download.pid"
    if pid_file.exists():
        old_pid = int(pid_file.read_text())
        if Path(f"/proc/{old_pid}").exists():
            return
    with (ROOT / "logs/download.log").open("ab") as file:
        process = subprocess.Popen([PYTHON, "-u", str(ROOT / "download_checkpoint.py")],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=file,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(str(process.pid))
    print(json.dumps({"event": "download_resumed", "pid": process.pid}), flush=True)


def main():
    (ROOT / "setup_runtime.pid").write_text(str(os.getpid()))
    env = os.environ.copy()
    env.update(UV_HTTP_TIMEOUT="90", UV_CONCURRENT_DOWNLOADS="1")
    cuda_libs = [str(path) for path in (ROOT / "deps/nvidia").glob("*/lib")]
    cuda_libs.append("/home/zhsu/jyluo/projects/openpi/.venv/lib/python3.11/site-packages/cusparselt/lib")
    env["LD_LIBRARY_PATH"] = ":".join(cuda_libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    command = ["/home/zhsu/.local/bin/uv", "pip", "install", "--python", PYTHON,
               "--target", str(ROOT / "deps"), "--no-deps", "--index-url",
               "https://download.pytorch.org/whl/cu128", "torch==2.7.1"]
    try:
        if not (ROOT / "deps/torch/version.py").exists():
            for attempt in range(3):
                print(json.dumps({"event": "install_attempt", "attempt": attempt + 1}), flush=True)
                result = subprocess.run(command, env=env, cwd=ROOT)
                if result.returncode == 0:
                    break
                time.sleep(10)
            else:
                raise RuntimeError("CUDA 12.8 PyTorch installation failed after three attempts")
        smoke = '''import sys,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'deps'))
import torch
from importlib import metadata
print('torch',torch.__version__,torch.version.cuda,flush=True)
assert torch.__version__.startswith('2.7.1+cu128')
x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16)
assert (x@x).float().sum().item()==4096
layer=torch.nn.Conv2d(3,8,3).cuda().to(torch.bfloat16)
with torch.inference_mode():
 y=layer(torch.zeros((1,3,32,32),device='cuda',dtype=torch.bfloat16))
assert torch.isfinite(y).all()
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
record={'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':metadata.version('transformers'),
        'gpu':torch.cuda.get_device_name(),'arch':torch.cuda.get_arch_list(),
        'kernel_matmul_smoke':'passed','kernel_conv_smoke':'passed','official_transformers_patch':'passed'}
(root/'logs/runtime_ready.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
'''
        env.update(CUDA_VISIBLE_DEVICES="0", JAX_PLATFORMS="cpu", TORCH_COMPILE_DISABLE="1")
        subprocess.run([PYTHON, "-u", "-c", smoke, str(ROOT)], env=env, cwd=ROOT, check=True)
    except Exception:
        message = traceback.format_exc()
        print(message, flush=True)
        with (ROOT / "logs/alerts.jsonl").open("a") as file:
            file.write(json.dumps({"time": time.time(), "event": "runtime_setup_failed", "error": message}) + "\n")
        raise
    finally:
        restart_download()


if __name__ == "__main__":
    main()
