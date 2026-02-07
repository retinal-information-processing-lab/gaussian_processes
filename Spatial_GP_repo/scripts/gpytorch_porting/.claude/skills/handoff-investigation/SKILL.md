---
name: handoff-investigation
description: Create a structured handoff document for investigation or debugging sessions. Use when an investigation hits a dead end, is deprioritized, or produces valuable findings worth preserving for future sessions. Also use when the user says "handoff" in the context of an investigation or exploration task. Produces a handoff file with findings, what was tried, and advice for future attempts.
---

# Investigation Handoff

Create a handoff document that captures what was learned during an investigation. Most commonly used when context is running out on an ongoing investigation — the work continues in a new session. Also used when an investigation is abandoned or paused indefinitely.

Before writing, ask the user: **"Are we continuing this investigation in a new session, or closing it for now?"** This determines the tone and the Status field (Continuing vs Paused/Abandoned).

## Step 1: Gather State

Run these commands and capture the output:
- `git branch --show-current`
- `git status`
- `git diff --stat`
- Check for investigation files in `investigations/` that relate to this work

## Step 2: Review Conversation

Scan the full conversation and extract:
- The problem being investigated and the original motivation
- Each approach tried, with its result and interpretation
- Key findings (with evidence — numbers, file paths, error messages)
- Why the investigation was stopped
- Observations that came up but were out of scope
- Honest points of confusion

## Step 3: Write Handoff File

Save to `investigations/[investigation-folder]/HANDOFF.md`. If the investigation folder already exists (from scripts created during the session), use it. If not, ask the user where to save.

Use this structure:

```
# Investigation: [Problem Title]

**Branch**: [branch name]
**Date**: [today]
**Status**: [Continuing / Paused / Abandoned / Inconclusive / Resolved - not implementing]
**Location**: `investigations/[folder]/`

---

## Problem Statement

[What was being investigated and why. Include the original motivation.]

## What Was Tried

### [Approach 1]
- **What**: [description]
- **Result**: [what happened, with numbers if applicable]
- **Interpretation**: [what this means]
- **Verdict**: [Dead end / Inconclusive — worth retrying / Promising — continue]

### [Approach 2]
...

## Key Findings

[Numbered list of things learned. Each finding must be
self-contained and include the evidence (file path, number, error).
Mark each as CONFIRMED (backed by data) or HYPOTHESIS (interpretation).]

## Why This Was Stopped

[Honest explanation. Context ran out, dead end, diminishing returns,
deprioritized, blocked by external factor, derailed, etc. No sugar-coating.
If status is "Continuing", this is just "context ran out" — focus on
where to pick up rather than why we stopped.]

## Things Noticed But Not Acted Upon

[Observations that came up during investigation but were
out of scope. Numbered, with enough context to evaluate later.]

## Uncommitted Changes

[git status output, annotated]

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| ... | ... | ... |

## If Someone Revisits This

[Practical advice for a future session.
- What to try next (most promising direction first)
- What NOT to try again (dead ends, with brief reason why)
- What prerequisite would make it tractable
The "what NOT to try" list prevents wasted time and is often
the most valuable part of the handoff.]

---

## Continuation Prompt

[Ready-to-paste block for a future session that picks this up.
10-20 lines. Contains: problem summary, pointer to this handoff file,
what was already tried, "check git status and git branch before starting".]
```

## Step 4: Post-Handoff Checklist

- Update CLAUDE.md if project status changed (e.g., move item to Deferred)
- Add a brief SESSION_LOG.md entry pointing to the handoff file (do not duplicate content):
  ```
  ## YYYY-MM-DD: [Brief Title]
  **Handoff**: `investigations/[folder]/HANDOFF.md`
  **Status**: [Abandoned / Paused / etc.]
  ```
- Print the continuation prompt to the conversation so the user can copy it
- Remind the user about uncommitted changes if any exist
- Recommend which investigation files to keep vs delete

## Quality Rules

- No emojis
- All file paths relative to project root
- Exact numbers -- no rounding or paraphrasing metrics
- "Why This Was Stopped" and "Key Findings" sections are mandatory
- The handoff must be readable by someone who was NOT in the session
- This handoff replaces the standard session wrap-up (working_guidelines.md Section 9, steps 1/2/4). Steps 3 and 5 still apply.
