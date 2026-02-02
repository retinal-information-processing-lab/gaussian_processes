# Investigation: Performance Loss with Increased ntrain and M

**Created by**: Claude (2026-02-01)
**Issue**: Some configurations show LOWER test_r with MORE training data or MORE inducing points

## Problem Statement

Historical benchmark data shows counterintuitive patterns:
1. **Increasing ntrain can DECREASE performance** (Cell 8: ntrain=500→2000 at M=200: test_r 0.81→0.74)
2. **Increasing M can DECREASE performance** (Cell 28: M=50→250: test_r 0.60→0.48)

This suggests potential overfitting, numerical issues, or optimization failures.

## Test Plan

**Cells**: 8, 10, 15 (8 is well-fitted, 15 is known hard case)
**Training sizes**: ntrain = [500, 2000]
**Inducing points**: M = [50, 100, 200]
**Modes**: vargp_old, vargp_direct (float32), default_gpy (adam, 500 iterations)
**Seed**: 123 (fixed for reproducibility)

## Files

- `run_systematic_test.py`: Main test script
- `results.csv`: Output results
- `ANALYSIS.md`: Findings and hypotheses
