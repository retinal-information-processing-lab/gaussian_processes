# Working Guidelines for Claude Code Sessions

**Purpose**: This document defines HOW Claude should work on this project. Read this BEFORE reading CLAUDE.md. These are meta-rules about process and philosophy, not project-specific technical details.

---

## 1. Core Philosophy

### 1.1 Scientist-Developer Balance
- The user is a scientist, not a professional developer
- Code must be comprehensible and modifiable by the user
- Think like both a software developer AND a scientist
- Scientific correctness matters more than software elegance

### 1.2 Simplicity Over Elegance
- **NO** explosion of unit tests, complicated classes, or abstract patterns
- **NO** unnecessary abstractions "for future flexibility"
- **YES** straightforward code that does one thing well
- If unsure, choose the simpler approach

### 1.3 First Version Mindset
- We are building a FIRST VERSION implementation
- Additional features can be added later if needed
- Don't optimize prematurely
- Don't add features that weren't requested

### 1.4 Precision Over Speed
- Take time to understand fully before implementing
- "ultrathink" - deep reflection before action
- Better to ask a clarifying question than to assume wrong
- Validate carefully before declaring something complete

---

## 2. Communication Rules

### 2.1 Push Back Respectfully
- Do NOT accept everything the user says as ground truth
- If a proposed approach is suboptimal, say so clearly
- Explain WHY with concrete reasoning
- Offer alternatives with tradeoffs
- But: don't overcomplicate the pushback itself

### 2.2 Ask Before Assuming
- Clarify scope and requirements early
- Don't assume what "better" means to the user
- When in doubt, ask - don't guess
- Questions are cheaper than wrong implementations

### 2.3 Explain Tradeoffs Clearly
- When presenting options, explain pros/cons
- Use concrete examples, not abstract principles
- Reference existing code when relevant
- Keep explanations concise

---

## 3. Development Process

### 3.1 Staged Implementation
- Break large tasks into stages
- Each stage should be independently testable
- Complete and validate Stage N before starting Stage N+1
- Example: Stage 1 (C=I) → Stage 2 (RF structure) → Stage 3 (custom E-step)

### 3.2 Verify Basic Case First
- Before adding complexity, verify the simple case works
- Use equivalence tests (e.g., new code matches reference with simple params)
- A working simple version is worth more than a broken complex version

### 3.3 Multiple Validation Layers
- Don't just test "does it run"
- Test equivalence with known-good reference
- Test gradient flow (for learnable params)
- Test actual performance improvement
- Example: C=I equivalence test + gradient test + Pearson r improvement test

### 3.4 Defer Complexity Explicitly
- When deciding NOT to implement something, document WHY
- Add it to a "Deferred Items" section
- Example: "Pixel masking - DEFERRED because adds complexity, not needed for first version"
- This prevents future sessions from re-debating the same questions

### 3.5 Always Test with Real Data
- **Default to real data** (PNAS dataset) for all validation tests
- **Ask before using synthetic data** - it can hide real-world issues
- The user assumes tests run on real data unless told otherwise

### 3.6 Include Timing in Test Scripts
- **Always add timing** to model fitting/training scripts
- Print elapsed time in results summary
- Enables performance comparisons across configurations

### 3.7 Flag debug code
- **Precede a code with "DEBUG" if you are writing it while debugging**
- Temporary / printing code should be easy to recognize
- Remove it when done

### 3.8 Log Benchmark Results Immediately

When running a **structured performance test** (not quick debugging):
1. Log results to `results/BENCHMARK_LOG.md` immediately after the run
2. Include: M, mode, explained variance, timing, gradient mode
3. Include the exact command used
4. Mark exploratory vs milestone results

**What to log vs skip:**
| Log it | Don't log it |
|--------|--------------|
| Intentional benchmark runs | Quick debug tests |
| Comparing implementations | Checking if code runs |
| Results that inform decisions | Exploratory iterations |

This prevents losing results during session compaction. Complements Section 5.5 (Reproducibility Rule).


## 4. Code Style

### 4.1 Simple, Readable Code
- Functions should be understandable without extensive comments
- Avoid clever tricks that require explanation
- Prefer explicit over implicit
- Use meaningful variable names that match the math

### 4.2 No Unnecessary Abstractions
- Don't create a class when a function will do
- Don't create a helper function for one-time operations
- Don't add configuration for hypothetical future needs
- Three similar lines is better than a premature abstraction

### 4.3 Match Reference Implementations
- New code should match behavior of existing known-good code
- Create explicit tests comparing new vs reference
- Document any intentional differences

### 4.4 Document Math-to-Code Mapping
- Include mathematical formulas in docstrings
- Map code variables to mathematical symbols
- Example: `# u = K_tilde_inv @ k (projection vector)`

---

## 5. Documentation Requirements

### 5.1 CLAUDE.md is Single Source of Truth
- All project knowledge, decisions, and state go in CLAUDE.md
- Do NOT scatter important info across multiple files
- Update CLAUDE.md immediately after completing work
- Future sessions should only need to read CLAUDE.md + this file

### 5.2 Decision Log Format
- Use Q&A format for design decisions
- Include: the question, the answer, the rationale
- Document alternatives considered and why rejected
- Example:
  ```
  **Q: Should we use RBF or arc-cosine kernel?**
  > A: Arc-cosine with C=I
  > Rationale: Keeps kernel math same while removing RF complexity
  > Alternative rejected: RBF (stationary, wouldn't test non-stationary integration)
  ```

### 5.3 Track Progress with Todo Lists
- Use TodoWrite for multi-step tasks
- Mark items complete immediately when done
- Keep the list current - remove stale items

### 5.4 No Useless Aesthetic Changes
- When editing files, only change what's needed for the task
- Do NOT change capitalization, formatting, or wording just for style
- Do NOT reformat tables or reorganize sections unless specifically requested
- Substantive multi-line changes are fine - just avoid cosmetic tweaks

### 5.5 Reproducibility Rule for Documented Results

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
   ```
   **Validation results** (from `test_estep_pnas.py --ntilde 25`):
   | M | Pearson r |
   | 25 | 0.62 |
   ```
3. **Include reproduction command** in the script's docstring or in CLAUDE.md

**Script requirements**:
- Self-contained (loads data, runs test, prints results)
- Configurable via command-line args for key parameters
- Prints the metrics that are documented
- Does NOT need to be a formal unit test - just reproducible

**Rationale**: Future sessions must be able to verify documented claims. Results without reproduction steps are technical debt that compounds across sessions.

---

## 6. Data Handling (General Rule)

- Never modify original datasets
- Keep optimized/modified images in separate variables
- Document data shapes and preprocessing steps

**Note**: Project-specific technical details (numerical precision, GPyTorch patterns, etc.) belong in CLAUDE.md, not here. This document is about process/philosophy, not project-specific knowledge.

---

## 7. Session Startup Checklist

When starting a new session on this project:

1. Read this file (WORKING_GUIDELINES.md) first
2. Read CLAUDE.md to understand project state
3. Check the "Implementation Stages" section for what's done/pending
4. Check the "Decision Log" for past design choices
5. Ask clarifying questions before implementing

---

## 8. Git Hygiene (Gentle Reminders)

The user is new to git. Provide occasional nudges, but don't make version control the focus.

### 8.1 When to Prompt About Git
- **After completing a milestone** (e.g., feature works, tests pass): "This might be a good point to commit"
- **Before major changes**: "Consider committing current working state first"
- **At session end** (if there's uncommitted work): Mention it briefly

### 8.2 Keep It Simple
- Don't lecture about git - just suggest the action
- Use simple commands: `git add -A && git commit -m "message"` then `git push`
- Offer to help with the commit message
- If something goes wrong with git, fix it without lengthy explanations
- If you take decisions about not including specific CODE-RELATED files, mention it clearly

### 8.3 Don't Overdo It
- One reminder per natural milestone is enough
- If the user ignores the suggestion, move on
- Git is a tool, not the goal

---

*Created: January 2025 (extracted from successful Stage 1-2 implementation)*
