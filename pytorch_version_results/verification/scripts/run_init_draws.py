"""PYTORCH_PARITY.md §7.6 driver: 6 TF-drawn (seeds 3-8) and 6 PyTorch-drawn (init=keras, seeds 4-9)
initial weight sets, all trained by experiment.py on PyTorch's seed-1 scenario order and z-scores,
no early stopping, each stopped as soon as it leaves the delay plateau (val_loss < 75 at an epoch
end) or after 25 epochs. Resumable: finished jobs are skipped, unfinished ones restart with --resume.

  setsid nohup /home/ubuntu/anaconda3/envs/RG_torch/bin/python results/verification/run_init_draws.py \
      > results/verification/logs/init_draws.log 2>&1 < /dev/null &
"""
import csv
import json
import os
import signal
import subprocess
import time

ROOT = "/home/ubuntu/code/RouteNet-Gauss-v2"
PY = "/home/ubuntu/anaconda3/envs/RG_torch/bin/python"
C = "trex_multiburst/RouteNetGauss/delay/seed_{s}"
ORDER = f"results/torch_converged_native/{C.format(s=1)}/sample_order_used.npy"
ZS = f"normalization/torch_converged_native/{C.format(s=1)}/z_scores.pkl"
STATE = "results/verification/init_draws/state.json"
EXIT_BELOW, MAX_EPOCHS, POLL_S, STALL_H = 75.0, 25, 30, 3.0
CONCURRENCY = int(os.environ.get("INITDRAW_CONCURRENCY", "2"))

JOBS = []
for tf_s, pt_s in zip(range(3, 9), range(4, 10)):  # interleaved so partial results stay balanced
    JOBS.append({"id": f"tf_s{tf_s}", "group": "TF-drawn", "seed": tf_s,
                 "init_weights": f"results/verification/init_draws/tf_seed_{tf_s}/init_weights.npz"})
    JOBS.append({"id": f"pt_s{pt_s}", "group": "PyTorch-drawn", "seed": pt_s, "init_weights": None})


def exp_name(job):
    return f"torch_initdraw_{job['id']}"


def cmd(job):
    c = [PY, "experiment.py", "--dataset", "trex_multiburst", "--target", "delay", "--seed", str(job["seed"]),
         "--epochs", str(MAX_EPOCHS), "--steps", "500", "--shuffle-buffer", "200", "--patience", "0",
         "--save-best-only", "--experiment-name", exp_name(job), "--device", "cuda", "--threads", "1",
         "--sample-order", ORDER, "--z-scores", ZS, "--resume"]
    if job["init_weights"]:
        c += ["--init-weights", job["init_weights"]]
    return c


def history(job):
    p = os.path.join("results", exp_name(job), C.format(s=job["seed"]), "history.csv")
    if not os.path.exists(p):
        return [], None
    rows = list(csv.DictReader(open(p)))
    return [float(r["val_loss"]) for r in rows], os.path.getmtime(p)


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def running_pid(job):
    """PID of an experiment.py already training this job (e.g. started by a previous driver)."""
    out = subprocess.run(["pgrep", "-f", "--", f"--experiment-name {exp_name(job)} "], capture_output=True, text=True).stdout
    for p in out.split():
        try:
            argv = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"experiment.py") for a in argv) and not any(a.endswith(b"bash") for a in argv[:1]):
            return int(p)
    return None


def alive(h):
    if isinstance(h, int):
        try:
            os.kill(h, 0)
            return True
        except ProcessLookupError:
            return False
    return h.poll() is None


def stop(h):
    pid = h if isinstance(h, int) else h.pid
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(120):
        if not alive(h):
            return
        time.sleep(1)


def main():
    os.chdir(ROOT)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    running = {}  # id -> (Popen, started)
    queue = [j for j in JOBS if state.get(j["id"], {}).get("status") != "done"]
    log(f"{len(JOBS) - len(queue)} jobs already done, {len(queue)} to run")
    while queue or running:
        # adopt jobs that are already training (a previous driver was replaced), then fill up
        for job in list(queue):
            pid = running_pid(job)
            if pid is not None:
                queue.remove(job)
                running[job["id"]] = (job, pid, time.time())
                log(f"ADOPT {job['id']} (pid {pid})")
        while queue and len(running) < CONCURRENCY:
            job = queue.pop(0)
            lf = open(f"results/verification/logs/initdraw_{job['id']}.log", "a")
            running[job["id"]] = (job, subprocess.Popen(cmd(job), stdout=lf, stderr=subprocess.STDOUT,
                                                        start_new_session=True), time.time())
            log(f"START {job['id']} ({job['group']}, seed {job['seed']})")
        time.sleep(POLL_S)
        for jid in list(running):
            job, proc, t0 = running[jid]
            vals, mtime = history(job)
            ex = next((i for i, v in enumerate(vals) if v < EXIT_BELOW), None)
            stalled = mtime is not None and (time.time() - mtime) / 3600 > STALL_H and alive(proc)
            if ex is not None or not alive(proc) or stalled:
                if alive(proc):
                    stop(proc)
                status = "done" if (ex is not None or len(vals) >= MAX_EPOCHS) else ("stalled" if stalled else "failed")
                state[jid] = {"group": job["group"], "seed": job["seed"], "status": status, "exit": ex,
                              "epochs": len(vals), "exit_val": vals[ex] if ex is not None else None,
                              "last_val": vals[-1] if vals else None, "hours": round((time.time() - t0) / 3600, 2)}
                json.dump(state, open(STATE, "w"), indent=2)
                log(f"END {jid}: {status}, plateau exit {ex if ex is not None else f'none in {len(vals)} epochs'}")
                del running[jid]
    log("ALL DONE")


if __name__ == "__main__":
    main()
