"""Finish only two fresh-model checks inside the original 30-minute diagnostic time bound."""
import csv
import json
import os
import signal
import time

from common import ROOT, OPENPI, write_json
import gpu_diagnostics as gpu


def main():
    # Wait for the original bounded queue, never run a duplicate model concurrently.
    while True:
        state=json.loads((ROOT/'artifacts/gpu_status.json').read_text())
        if state['state']=='completed':
            break
        if state['state'] in ('blocked','stopped_by_signal'):
            raise RuntimeError('original queue stopped; do not expand GPU work')
        time.sleep(2)
    with (ROOT/'artifacts/gpu_samples.csv').open() as file:
        first=float(next(csv.reader(file))[0])
    elapsed=time.time()-first
    gpu.STARTED=time.monotonic()-elapsed
    signal.signal(signal.SIGTERM,gpu.stop)
    signal.signal(signal.SIGINT,gpu.stop)
    try:
        for tag in ('fixed_reload_1','fixed_reload_2'):
            if (ROOT/'artifacts'/f'{tag}.json').exists():
                continue
            gpu_id=gpu.wait_gpu()
            if time.time()-first>1680:
                raise TimeoutError('insufficient remaining original GPU diagnostic budget for reload')
            child=gpu.launch([str(OPENPI/'.venv/bin/python'),'-u',str(ROOT/'fixed_input_reload.py'),'--tag',tag],gpu_id,tag)
            gpu.wait(child,gpu_id)
        write_json(ROOT/'artifacts/gpu_status.json',{'state':'completed','pid':os.getpid(),'training_updates':0,'followup':'two fresh-process fixed-input checks completed','gpu_elapsed_seconds':time.time()-first})
    finally:
        for child in gpu.CHILDREN:
            gpu.helpers.terminate(child)


if __name__=='__main__':
    main()
