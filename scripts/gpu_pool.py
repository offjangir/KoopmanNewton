"""Pack many small single-GPU jobs onto the local GPUs.

Models here are tiny (3-90M) and a GH200 has headroom for several at once, so each
GPU runs up to ``per_gpu`` jobs. A job is ``(cmd, log_path)``; it is pinned to one
device through ``CUDA_VISIBLE_DEVICES`` composed with the allocation's own mask
(overwriting the mask outright makes jobs fail in ``_cuda_init`` whenever the
allocation exposes a different physical device set).
"""

from __future__ import annotations

import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_pool(jobs, n_gpu: int = 4, per_gpu: int = 3, threads: int = 2, poll: float = 5.0):
    """Run ``[(cmd, log_path), ...]`` from the repo root. Returns the failed jobs."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = vis.split(",") if vis else [str(g) for g in range(n_gpu)]
    queue, running, load, failed = list(jobs), {}, [0] * n_gpu, []
    done, t0 = 0, time.time()

    def launch(job):
        cmd, log_path = job
        g = min(range(n_gpu), key=lambda i: load[i])
        load[g] += 1
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[g],
                   OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        log = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        running[proc] = (job, g, log, time.time())

    print(f"{len(queue)} jobs on {n_gpu} gpus x {per_gpu}", flush=True)
    while queue and len(running) < n_gpu * per_gpu:
        launch(queue.pop(0))
    while running:
        time.sleep(poll)
        for proc in list(running):
            if proc.poll() is None:
                continue
            job, g, log, ts = running.pop(proc)
            log.close()
            load[g] -= 1
            done += 1
            if proc.returncode != 0:
                failed.append(job)
            status = "ok  " if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
            print(f"[{done}/{done + len(running) + len(queue)} {status} gpu{g} "
                  f"{time.time() - ts:4.0f}s] {job[1]}", flush=True)
            if queue:
                launch(queue.pop(0))
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed", flush=True)
    return failed
