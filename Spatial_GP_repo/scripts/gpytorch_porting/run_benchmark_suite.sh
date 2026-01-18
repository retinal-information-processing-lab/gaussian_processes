#!/bin/bash
# Benchmark suite for GPyTorch porting project
# Run with: nohup bash run_benchmark_suite.sh > benchmark_run.log 2>&1 &
# Check progress: tail -f benchmark_run.log
# Results will be in: results/benchmark_2026-01-18/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="results/benchmark_2026-01-18"
mkdir -p "$RESULTS_DIR"

echo "=== Benchmark Suite Started: $(date) ==="
echo "Results will be saved to: $RESULTS_DIR"
echo ""

run_test() {
    local mode=$1
    local gradient=$2
    local ntilde=$3
    local outfile="$RESULTS_DIR/${mode}_${gradient}_M${ntilde}.log"

    echo ">>> Running: mode=$mode, gradient=$gradient, M=$ntilde"
    echo "    Output: $outfile"

    if [ "$gradient" == "none" ]; then
        # vargp_old doesn't use gradient-mode flag
        conda run -n pytorch_gpytorch python test_estep_pnas.py \
            --mode "$mode" --ntilde "$ntilde" 2>&1 | tail -100 > "$outfile"
    else
        conda run -n pytorch_gpytorch python test_estep_pnas.py \
            --mode "$mode" --gradient-mode "$gradient" --ntilde "$ntilde" 2>&1 | tail -100 > "$outfile"
    fi

    echo "    Done: $(date)"
    echo ""
}

# === vargp_old (reference, no gradient-mode) ===
echo "=== vargp_old (reference) ==="
for M in 50 75 100 200; do
    run_test vargp_old none $M
done

# === vargp_style + autograd ===
echo "=== vargp_style + autograd ==="
for M in 50 75 100 200; do
    run_test vargp_style autograd $M
done

# === vargp_style + vjp ===
echo "=== vargp_style + vjp ==="
for M in 50 75 100 200; do
    run_test vargp_style vjp $M
done

# === adam + autograd ===
echo "=== adam + autograd ==="
for M in 50 75 100 200; do
    run_test adam autograd $M
done

# === adam + vjp ===
echo "=== adam + vjp ==="
for M in 50 75 100 200; do
    run_test adam vjp $M
done

echo "=== All benchmarks complete: $(date) ==="
echo ""
echo "=== SUMMARY ==="
echo "Extracting results from log files..."
echo ""

# Extract results into a summary file
SUMMARY="$RESULTS_DIR/SUMMARY.txt"
echo "Benchmark Summary - $(date)" > "$SUMMARY"
echo "================================" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for logfile in "$RESULTS_DIR"/*.log; do
    if [ -f "$logfile" ]; then
        fname=$(basename "$logfile")
        echo "--- $fname ---" >> "$SUMMARY"
        grep -E "(RESULTS:|Training time:|Explained var:|Test Pearson)" "$logfile" >> "$SUMMARY" 2>/dev/null || echo "  (no results found)" >> "$SUMMARY"
        echo "" >> "$SUMMARY"
    fi
done

echo "Summary saved to: $SUMMARY"
cat "$SUMMARY"
