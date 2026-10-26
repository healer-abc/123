# Active experiment

> SUPERSEDED 2026-10-15 20:43 CST: user replaced shelf-place with peg-insert-side and requested an unfinetuned base checkpoint. Old watchdog/bootstrap/setup/download PIDs stopped after exact command-line verification. No formal old SFT episodes completed. Do not restart this old pipeline; current experiment is ../pi05_metaworld_base_20261015/.

User requested exactly 20 frozen pi0 rollouts for each of assembly-v2, shelf-place-v2,
disassemble-v2, peg-unplug-side-v2, stick-pull-v2, with a final success-rate table.
Do not improve or train the model. User explicitly selected the official RLinf
MetaWorld SFT checkpoint after initially considering retraining.

This workspace is the same server reached by `ssh zhsu@dzhou.ltd -p 7005` (aius-04).
Files under this directory can be accessed directly. Do not modify the existing
openpi environment or stop its two running pi05 LIBERO training processes.

`protocol.json` fixes the checkpoint, native MetaWorld v2 commit, task and policy
seeds, 15 settling steps, 160 policy steps, action chunk 5 and denoising steps 5.
Results use environment success_once and also record last-step success. Runtime
errors must not become task failures. No random/expert controller is a baseline.

Background lifecycle (now managed by systemd user services, not chat-launched jobs):

- `pi0-mw-runtime.service`: dependency installation and CUDA validation.
- `pi0-mw-watchdog.service`: persistent supervisor and resumable download recovery.
- `pi0-mw-bootstrap.service`: actual policy smoke validation and release of the full evaluation.

Inspect with `systemctl --user status pi0-mw-runtime pi0-mw-watchdog pi0-mw-bootstrap`.
All have `Restart=on-failure`; logs still append under this experiment directory.
Prior directly launched preparation processes all exited around the same time;
their exact termination cause is unconfirmed. Do not rely on old PID files without
checking both process existence and fresh progress. The replacement service cgroups
are under `user@1009.service/app.slice`, separate from the chat execution session.

1. `setup_runtime.py` installs torch==2.7.1+cu128 into `deps/`, runs real CUDA BF16
   matmul/convolution and official Transformers patch checks, then creates
   `logs/runtime_ready.json`. It always resumes `download_checkpoint.py` afterward.
2. `download_checkpoint.py` resumes part files and verifies the ModelScope
   publisher's full SHA256 before writing `verified.json`. The official mirror
   contains a different revision than HF main; preserve the documented MS hash.
3. `bootstrap.py` waits for both assets and runtime, runs one excluded 160-step
   model rollout, then creates `logs/policy_smoke_passed.json` only on success.
4. `watchdog.py` then runs up to two workers, one per GPU with at least 16 GB free.
   It saves telemetry/alerts, retries only transient failures with identical seeds,
   and calls `aggregate.py` to update CSV/JSON/Markdown results.

Read `logs/watchdog_state.json`, `logs/setup_runtime.log`, `logs/download.log`,
`logs/bootstrap.log`, `logs/policy_smoke.log`, and `logs/alerts.jsonl` for actual
status. PID files are in this directory. Verify `/proc/PID/stat` is not zombie.
Do not report an experiment as running/completed just because a watchdog exists.
Formal result files are `results/TASK/episode_XX.json`; smoke files are excluded.

Completed validation: all 5 native v2 tasks reset, accept 4D actions and render
480x480 RGB with official camera settings. PNGs are under `videos/`. CPU meta-device
model audit found no checkpoint shape mismatches or unexpected parameters; one
missing name is a tied embedding alias, to be checked by real safetensors loading.

The shared environment has torch 2.7.1+cu126, which fails real sm_120 GPU operations.
Transformers 4.53.2 is copied into isolated deps and the publisher's replacement
files have been applied there. `deps/nvidia` is a read-only reuse via symlink of
existing CUDA 12.8 shared libraries in the gemma-lora environment. Validate the
runtime before allowing the full evaluation. Existing training uses JAX and remains
untouched.

Network observation: many simultaneous large weight requests caused other HTTPS
requests to time out. Temporarily pausing only our weight transfer allowed small
dependency/source downloads to finish. The resumable SFT download now runs with
two concurrent transfers while torch installs, expanding to twelve only after the
CUDA runtime validation flag exists. The supervisor retries a dead/stalled downloader
up to three times while preserving part files. The final result must clearly label
v2 results as an adapted evaluation, not a reproduction of the MT50 v3 score.
