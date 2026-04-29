"""NGD validation pipeline: B → C → A (in order).

Runs the three experiments sequentially. Each is crash-safe and resumes
from its own results JSONL. Safe to kill and rerun.
"""
import subprocess, sys, time
from pathlib import Path

DIR = Path(__file__).parent
PYTHON = sys.executable

experiments = [
    ('B', 'run_B_freeamp_64x64.py',   'NGD free Amp 64x64        (41 runs, ~10 min)'),
    ('C', 'run_C_lowntrain_64x64.py',  'Low n_train 64x64 both    (246 runs, ~2 h)'),
    ('A', 'run_A_108x108.py',          'NGD 108x108               (41 runs, ~4 h)'),
]

print("NGD validation pipeline starting", flush=True)
wall_total = time.time()

for tag, script, desc in experiments:
    print(f"\n{'='*60}", flush=True)
    print(f"  Starting Exp {tag}: {desc}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    ret = subprocess.run([PYTHON, str(DIR / script)])
    elapsed = time.time() - t0
    status = 'OK' if ret.returncode == 0 else f'FAILED (rc={ret.returncode})'
    print(f"\n  Exp {tag} finished: {status}  in {elapsed/60:.1f} min", flush=True)
    if ret.returncode != 0:
        print(f"  Pipeline stopping on Exp {tag} failure.", flush=True)
        sys.exit(ret.returncode)

print(f"\n{'='*60}", flush=True)
print(f"  All experiments done in {(time.time()-wall_total)/3600:.2f} h", flush=True)
print(f"{'='*60}", flush=True)
