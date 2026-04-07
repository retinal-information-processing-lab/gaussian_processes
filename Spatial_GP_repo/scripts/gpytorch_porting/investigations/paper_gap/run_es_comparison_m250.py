"""Early stopping comparison at M=250, 64x64, 3 seeds, 5 representative cells.

Uses the best sweep config equivalent: interleaved F-step, fix_Amp, A_init=1e-4, beta=0.1.
Early stopping enabled with default patience=15, min_delta_rel=0.001.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from run_single_mode import build_config_from_defaults, run_single_config

DATA_64 = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_CENTERS = np.load(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'rf_centers_ground_truth.npz'), allow_pickle=True)

CELLS = [0, 1, 8, 18, 28]
SEEDS = [1, 2, 3]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'es_comparison_m250_results.jsonl')

def main():
    total = len(CELLS) * len(SEEDS)
    print(f"=== Early Stopping Comparison (M=250, 64x64) ===")
    print(f"  {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Results: {RESULTS_FILE}")
    
    results = []
    for cell_id in CELLS:
        for seed in SEEDS:
            eps_0x, eps_0y = RF_CENTERS['norm_64'][cell_id]
            config = build_config_from_defaults(
                mode='vargp_direct',
                M=250,
                n_train=3160,  # will be capped to 2910 (val held out)
                seed=seed,
                cell=cell_id,
                data_path=DATA_64,
                eps_0x=float(eps_0x),
                eps_0y=float(eps_0y),
                beta=0.1,
                rho=0.1,
                A_init=1e-4,
                lambda0_init=-1.0,
                n_iterations=80,
                n_estep=50,
                n_mstep=20,
            )
            config['early_stop'] = True
            config['ip_selection'] = 'random'
            config['fix_Amp'] = True
            config['interleave_fstep'] = True
            
            t0 = time.time()
            result = run_single_config(config)
            wall_time = time.time() - t0
            
            result['wall_time'] = wall_time
            result['config_name'] = 'es_m250_intl_fixAmp'
            results.append(result)
            
            with open(RESULTS_FILE, 'a') as f:
                # Sanitize for JSON
                import torch
                def sanitize(v):
                    if isinstance(v, torch.Tensor):
                        return v.detach().cpu().tolist() if v.numel() > 1 else v.item()
                    if isinstance(v, (np.floating, np.integer)):
                        return v.item()
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                    if isinstance(v, dict):
                        return {k: sanitize(vv) for k, vv in v.items()}
                    if isinstance(v, list):
                        return [sanitize(vv) for vv in v]
                    if isinstance(v, (str, int, float, bool, type(None))):
                        return v
                    return str(type(v).__name__)  # skip non-serializable objects
                f.write(json.dumps(sanitize(result)) + '\n')
            
            stopped = result.get('stopped_early', False)
            best_it = result.get('best_iteration', '?')
            final_it = result.get('final_iteration', '?')
            print(f"  cell={cell_id} seed={seed}: test_r={result['test_r']:.4f} expl_var={result['explained_var']:.4f} "
                  f"time={wall_time:.1f}s iters={final_it} best={best_it} stopped={stopped}")
    
    # Summary
    print(f"\n=== Summary (mean over seeds) ===")
    for cell_id in CELLS:
        cr = [r for r in results if r['cell'] == cell_id]
        tr = np.mean([r['test_r'] for r in cr])
        ev = np.mean([r['explained_var'] for r in cr])
        t = np.mean([r['wall_time'] for r in cr])
        fi = np.mean([r.get('final_iteration', 0) for r in cr])
        bi = np.mean([r.get('best_iteration', 0) for r in cr])
        print(f"  Cell {cell_id:2d}: test_r={tr:.4f} expl_var={ev:.4f} time={t:.1f}s iters={fi:.0f} best={bi:.0f}")

if __name__ == '__main__':
    main()
