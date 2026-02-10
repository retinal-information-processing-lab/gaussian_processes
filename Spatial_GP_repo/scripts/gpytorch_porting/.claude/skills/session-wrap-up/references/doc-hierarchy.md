# Documentation Hierarchy

This maps all documentation in the project and what each file covers. Use this to identify which docs need updating after a session.

## Tier 1: Always Loaded (rules, no path restriction)

| File | Covers |
|------|--------|
| `.claude/rules/working_guidelines.md` | Process, philosophy, communication rules, dev process, git hygiene, session wrap-up |
| `.claude/rules/critical_short_rules.md` | Coordinates, variational params, import side effects, eigenspace, float32, config API, numerical debugging |

## Tier 2: Path-Matched Rules (auto-load when touching relevant files)

| File | Triggers on | Covers |
|------|------------|--------|
| `.claude/rules/math.md` | Math/formula files | Variational GP math, E-step derivations, KL divergence |
| `.claude/rules/gradients.md` | Gradient files | Analytical kernel gradient formulas |
| `.claude/rules/acquisition.md` | acquisition.py, test_acquisition.py | Utility functions, entropy, conditioning, A/lambda0 transform |
| `.claude/rules/debugging.md` | Test files | Known issues, torch.pi workaround, debugging patterns |
| `.claude/rules/jitter.md` | Jitter-related files | Cholesky stability, GPyTorch jitter internals, config wiring |

## Tier 3: Reference Files (read on demand, never auto-loaded)

| File | Covers |
|------|--------|
| `.claude/EIGENSPACE_REFERENCE.md` | vargp_direct implementation, eigenspace projection |
| `.claude/PATTERNS_REFERENCE.md` | GPyTorch code patterns |
| `.claude/DATA_REFERENCE.md` | Data format, preprocessing, dataset loader |
| `.claude/DECISION_LOG.md` | Design rationale (Q&A format, Q1-Q25+) |

## Tier 4: Main Project Doc

| File | Sections to check |
|------|------------------|
| `.claude/CLAUDE.md` | Quick Start table, Current Status table, CRITICAL RULES, Parameter Matching Table, File Map, Deferred Items, Known Issues |

## Tier 5: Session Timeline

| File | Format |
|------|--------|
| `SESSION_LOG.md` | Chronological entries, one per session |

## When to Create New Documentation

**Create a REFERENCE file** (.claude/*_REFERENCE.md) when:
- A problem area was investigated across multiple sessions and is now solved/stable
- The solution has non-obvious architecture worth documenting for future modification
- Examples: EIGENSPACE_REFERENCE (eigenspace math), PATTERNS_REFERENCE (GPyTorch patterns)

**Create a rule** (.claude/rules/*.md with path matching) when:
- There are active gotchas that Claude WILL get wrong without the rule in context
- The area involves specific files that can trigger the rule via path matching
- Examples: jitter.md (loaded when touching gpy_training.py), acquisition.md (loaded for acquisition.py)

**Neither is needed** when:
- The work is a straightforward bug fix or feature addition
- The changes are self-explanatory from code + commit message
- Existing docs already cover the area adequately
