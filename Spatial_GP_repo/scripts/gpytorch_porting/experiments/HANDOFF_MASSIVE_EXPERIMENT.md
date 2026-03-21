# Massive All-Cells Experiment — Handoff for Monitoring Sessions

**Created**: 2026-03-20
**Branch**: pietro/workingbranch
**Working dir**: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting`

## What's Running

Two experiments fitting GP models for all 41 PNAS cells, running sequentially (108x108 first, then 64x64):

| Experiment | Dataset | Runs | Status |
|------------|---------|------|--------|
| `massive_allcells_108` | 108x108 (11,664 px) | 492 | running first |
| `massive_allcells_64` | 64x64 (4,096 px) | 492 | queued after 108 |

**Matrix per experiment**: 41 cells x 3 seeds (1,2,3) x 2 modes (vargp_direct, default_gpy) x 2 M values (300, 2910) = 492

**Execution order** (itertools.product): all M=300 runs complete before M=2910.

## Quick Status Check

```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting

# 1. Is the process alive?
cat experiments/2026-03-20_massive_allcells_108/run.pid 2>/dev/null && \
  ps -p $(cat experiments/2026-03-20_massive_allcells_108/run.pid) -o pid,etime,cmd --no-headers

# 2. Progress count (expect 492 per experiment)
echo "108x108:" && wc -l experiments/2026-03-20_massive_allcells_108/results.jsonl 2>/dev/null
echo "64x64:" && wc -l experiments/2026-03-20_massive_allcells_64/results.jsonl 2>/dev/null

# 3. Failures
echo "108 errors:" && grep -c '"status": "error"' experiments/2026-03-20_massive_allcells_108/results.jsonl 2>/dev/null
echo "64 errors:" && grep -c '"status": "error"' experiments/2026-03-20_massive_allcells_64/results.jsonl 2>/dev/null

# 4. Recent log output
tail -30 experiments/2026-03-20_massive_allcells_108/run.log

# 5. Detailed analysis
python analyze_experiment.py --exp massive_allcells_108
python analyze_experiment.py --exp massive_allcells_64
```

## Estimating Remaining Time

```bash
# Average time per M value (from results.jsonl)
python -c "
import json
for exp in ['massive_allcells_108', 'massive_allcells_64']:
    path = f'experiments/2026-03-20_{exp}/results.jsonl'
    try:
        records = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        print(f'{exp}: not started yet'); continue
    success = [r for r in records if r.get('status') == 'success']
    for M in [300, 2910]:
        subset = [r for r in success if r['M'] == M]
        if subset:
            avg_t = sum(r['time_total_s'] for r in subset) / len(subset)
            done = len(subset)
            total = 246  # 41 cells x 3 seeds x 2 modes
            remaining = total - done
            print(f'{exp} M={M}: {done}/{total} done, avg={avg_t:.1f}s/run, ~{remaining*avg_t/3600:.1f}h remaining')
        else:
            print(f'{exp} M={M}: 0/246 done')
"
```

## If Process Died — Resume

```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting

nohup bash -c '
  cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting
  echo "=== Resuming at $(date) ==="
  python run_experiment.py --exp massive_allcells_108 --resume
  python run_experiment.py --exp massive_allcells_64 --resume
  echo "=== All done at $(date) ==="
' >> experiments/2026-03-20_massive_allcells_108/run.log 2>&1 &

echo $! > experiments/2026-03-20_massive_allcells_108/run.pid
echo "Resumed with PID: $!"
```

## Experiment Parameters

- **n_train**: 2910 (all training images)
- **M (inducing points)**: 300 and 2910
- **Seeds**: 1, 2, 3
- **Kernel**: arc_cosine (default params from canonical.yaml)
- **n_iterations**: 50 (with early stopping: window=20, threshold=0.005)
- **Inducing selection**: pivoted Cholesky
- **dtype**: float32, device: cuda

## Smoke Tests

Before the full run, smoke tests were executed:
- `experiments/2026-03-20_smoke_108/` (8 runs: cells [0,40], seed 1, n_iter=3)
- `experiments/2026-03-20_smoke_64/` (8 runs: cells [0,40], seed 1, n_iter=3)

Check smoke test results: `python analyze_experiment.py --exp smoke_108`

## File Locations

```
experiments/
  HANDOFF_MASSIVE_EXPERIMENT.md          <- this file
  2026-03-20_massive_allcells_108/
    config.yaml                          <- frozen experiment config
    metadata.yaml                        <- git commit, timestamp
    results.jsonl                        <- one JSON record per completed run
    run.log                              <- stdout/stderr from nohup
    run.pid                              <- PID of background process
  2026-03-20_massive_allcells_64/
    (same structure, created when 64x64 starts)
  2026-03-20_smoke_108/                  <- smoke test results
  2026-03-20_smoke_64/                   <- smoke test results
```
