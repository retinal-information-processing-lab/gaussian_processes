---
paths:
  - "tests/**/*.py"
  - "run_*.py"
  - "investigations/**/*"
---

# Debugging, Testing & Reproducibility Rules

**Auto-loads when**: Working with test files, run scripts, or investigations.
**Manual invocation**: `/debug` skill

---

## 1. Quick Test Commands

```bash
# Single test (quick check)
python run_single_mode.py --mode vargp_style --explicit-unwhitening --seed 123

# Canonical test matrix (12 configs, regression testing)
python run_canonical_tests.py --seed 123

# Query benchmark results
python query_benchmark.py --mode vargp_style --M 50
python query_benchmark.py --compare-seeds 123 456

# vargp_direct mode (MUST use --float32)
python run_single_mode.py --mode vargp_direct --float32 --ntilde 50 --seed 123
```

**GPU REQUIRED**: Scripts default to CUDA. CPU is too slow.
Raise to user if unavailable.
---

## 2. Debugging Workflow (Step-by-Step)

### 2.1 Verify Basic Case First
- Before debugging complex configs, test the simplest version
- Use equivalence tests (e.g., new code matches reference with simple params)
- Example: C=I test before adding RF structure

### 2.2 Multiple Validation Layers
Don't just test "does it run":
1. **Does it run?** No errors
2. **Equivalence test**: Match known-good reference
3. **Gradient test**: For learnable params
4. **Performance test**: Pearson r improvement

### 2.3 Always Test with Real Data
- **Default to real data** (PNAS dataset) for all validation
- **Ask before using synthetic data** - it can hide real-world issues. DISCOURAGED
- User assumes tests run on real data unless told otherwise

### 2.4 Include Timing in Test Scripts
- **Always add timing** to model fitting/training scripts
- Print elapsed time in results summary
- Enables performance comparisons across configurations

### 2.5 Flag Debug Code
- **Precede code with "DEBUG"** if writing while debugging
- Temporary/printing code should be easy to recognize
- Remove it when done

### 2.6 Note Tolerance Increases
- If you increase numerical tolerance to make test pass, **justify and document it**
- **Mention to user** - don't silently relax tolerances

### 2.7 Prefer running known script to inline python
- Limit the use of inline python scripts that load data, tests should be done with reals
scripts in a dedicated /investigation folder

---

## 3. Known Issues & Workarounds

### 3.1 torch.pi Workaround (HACKY)
`tests/test_utils.py:set_reproducible_seed()` contains:
```python
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # WHY DOES THIS MATTER?!
```
**Workaround**: Replicates side effect from `GP_utils.py` line 49. Root cause unknown.

### 3.2 Jitter Consistency
**Issue**: All jitter values MUST match `model.jitter` (default 1e-4).
**Fix**: All `estep.py` functions now default to `model.jitter`.
**Symptom**: Mismatch causes whitening failures.

### 3.3 set_reproducible_seed Device Parameter
**Issue**: `set_reproducible_seed(seed, device=device)` produces DIFFERENT random sequences than `set_reproducible_seed(seed)`.
**Workaround**: Be consistent - always use same device parameter. Root cause unknown.

### 3.4 Whitening Seed Sensitivity
**Issue**: Whitened mode can be sensitive to random seed in some configurations.
**Debug approach**: Try multiple seeds to isolate seed-specific failures.

### 3.5 RF Center Initialization
**Issue**: RF center (eps_0x, eps_0y) needs reasonable init near image center.
**Symptom**: Won't learn from bad init.
**Fix**: Initialize near image center (e.g., 54, 54 for 108×108 images).

### 3.6 default_gpy Mode Fails on Some Cells
**Issue**: Cell 10 (ntrain=2000) unlearnable with default_gpy: test_r ≈ -0.05 to 0.11 vs vargp_direct ≈ 0.90.
**Root cause**: Adam optimizer cannot optimize this problem even with 500 iterations.
**Status**: Waiting for fix to try.

### 3.7 Performance Degradation with Large ntrain+M
**Issue**: Both vargp_direct and vargp_old show performance loss when ntrain=2000 and M≥100 on some cells.
**Example**: Cell 8: -10% loss.
**Status**: Cell-dependent issue under investigation.
**See**: `investigations/performance_loss_ntrain_M/`

### 3.8 Variational Strategy & Whitening Confusion
**Two distinct concepts**:

1. **standard_variational_distribution** (model.py):
   - `True` (default): Use `VariationalStrategy` (whitened params)
   - `False`: Use `UnwhitenedVariationalStrategy` (natural params)
   - CLI: `--unwhitened-variational-dist`

2. **explicit_unwhitening** (train.py/estep.py):
   - Controls L_K whitening conversions in E-step
   - **REQUIRED** for `vargp_style` with standard distribution
   - CLI: `--explicit-unwhitening` or `--no-explicit-unwhitening`

**CLI examples**:
```bash
# Standard distribution with explicit unwhitening (most common)
python run_single_mode.py --mode vargp_style --explicit-unwhitening

# Unwhitened strategy
python run_single_mode.py --mode vargp_style --unwhitened-variational-dist --no-explicit-unwhitening
```

---

## 4. Comparison & Reproducibility

### 4.1 Benchmark System (JSONL-based)

**Primary output**: `results/benchmark_results.jsonl` (machine-readable, append-only)
**Legacy (frozen)**: `results/BENCHMARK_LOG.md` (historical, do not update)

**Recording results**:
```bash
# Single test with JSON output
python run_single_mode.py --mode vargp_style --ntilde 50 --seed 123 --json-append results/benchmark_results.jsonl

# Run canonical test matrix (12 configs)
python run_canonical_tests.py --seed 123

# Query results
python query_benchmark.py --mode vargp_style --M 50
python query_benchmark.py --compare-seeds 123 456
```

**Before recording**:
- Check working tree is clean (`git status`)
- If uncommitted changes exist, **remind user to commit first** - results must be reproducible
- Commit hash is recorded automatically in JSON

**What to log vs skip:**

| Log it | Don't log it |
|--------|--------------|
| Intentional benchmark runs | Quick debug tests |
| Comparing implementations | Checking if code runs |
| Results that inform decisions | Exploratory iterations |

### 4.2 Canonical Test Matrix

Standard configurations for regression testing (12 per seed):

| ntrain | M values | Modes |
|--------|----------|-------|
| 2000 | 200 |  vargp_direct, gpytorch_defauls |
| 500 | 50, 100, 200 |  vargp_direct, gpytorch_default|

**When to run**:
- After completing a feature (regression check)
- After merge (verify no breakage)
- When comparing implementations

**What is NOT canonical** (don't log to main JSONL):
- Quick debug tests during development
- Exploratory parameter sweeps
- Checking if code runs

For exploratory work, use `--json-append results/exploratory.jsonl` instead.

### 4.3 Reproducibility Rule for Documented Results

**Core Rule**: If experimental results are important enough to document in CLAUDE.md, they MUST be reproducible via a script.

**When this applies**:
- Validation results with specific numbers (e.g., "Pearson r = 0.62 with M=25")
- Comparison tables between implementations
- Debugging findings that inform design decisions
- Any quantitative claim that future sessions might need to verify

**When this does NOT apply**:
- Quick exploratory tests during debugging (not documented)
- Results that are immediately superseded
- Conceptual observations without specific numbers

**Required actions**:
1. **Create a test script** in the project directory (e.g., `test_estep_pnas.py`)
2. **Reference the script in CLAUDE.md** next to the documented results:
   ```markdown
   **Validation results** (from `test_estep_pnas.py --ntilde 25`):
   | M | Pearson r |
   | 25 | 0.62 |
   ```
3. **Include reproduction command** in the script's docstring or in CLAUDE.md

**Script requirements**:
- **NO RANDOM ELEMENTS** - Seeds are controlled, same code execution → EXACTLY THE SAME RESULTS
- Self-contained (loads data, runs test, prints results)
- Configurable via command-line args for key parameters
- Prints the metrics that are documented
- Does NOT need to be a formal unit test - just reproducible

**Rationale**: Future sessions must be able to verify documented claims. Results without reproduction steps are technical debt.

### 4.4 Tests are Run from Scripts

When user says to run the test they mean using one of the dedicated scripts.

Full cell fits using inline python scripts are reserved for debugging.

---

## 5. Bug Investigation Cleanup

Sessions dedicated to codebase exploration or bug investigation require TIDYNESS:

1. **Use a dedicated folder** with explicit name in `gpytorch_porting/investigations/` path
2. **Keep track of files** you create and if they need to be cleaned up afterwards
3. **Document findings** in investigation folder README or CLAUDE.md
4. **Archive when done** - don't leave orphaned debug files

**Example structure**:
```
investigations/
├── performance_loss_ntrain_M/
│   ├── README.md           # What we found
│   ├── test_cell8_M100.py  # Reproducible test
│   └── results.txt         # Raw data
└── whitening_seed_issue/
    └── ...
```

---

## 6. Conda Environment Pattern

**The conda environment `pytorch_gpytorch` is already active.** Do NOT try to activate it.

**Correct pattern** (just use python directly):
```bash
python run_single_mode.py --mode vargp_direct --float32 --seed 123
```

**Patterns that FAIL** (don't use these):
```bash
# FAILS - conda activate requires shell initialization
source ~/.bashrc && conda activate pytorch_gpytorch && python script.py

# FAILS - same issue
conda activate pytorch_gpytorch && python script.py

# FAILS - wrong path (anaconda3, not miniconda3)
/home/idv-eqs8-pza/miniconda3/envs/pytorch_gpytorch/bin/python script.py
```

**Why**: The terminal session starts with `pytorch_gpytorch` already activated. Attempting to re-activate causes errors because bash isn't initialized for conda hooks in non-interactive mode.

**If you need to verify**: Run `which python` - should show `/home/idv-eqs8-pza/anaconda3/envs/pytorch_gpytorch/bin/python`.

---

## 7. Common Paths Followed (Accumulated Bug-Solving Wisdom)

**Purpose**: Document non-obvious debugging solutions. Add entries when you solve a tricky bug. Wrong turns are informative too

**Format**: `[Date] Symptom → Root cause → Solution (files: ...)`

---

### Entries

**2025-02** vargp_direct test_r degrades after iteration 20 → Eigenvalue tolerance too strict (1e-4) → Lower to 1e-3 OR use early stopping  ( INFORM USER IF DOING THIS)

---

**Instructions**: When you solve a non-trivial bug, add a one-line entry above. Include date, symptom, cause, fix, key files. User will reorganize periodically.
