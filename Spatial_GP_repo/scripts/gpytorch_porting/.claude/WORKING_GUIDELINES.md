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

---

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

*Created: January 2025 (extracted from successful Stage 1-2 implementation)*
