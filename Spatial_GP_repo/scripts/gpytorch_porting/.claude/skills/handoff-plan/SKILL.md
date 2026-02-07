---
name: handoff-plan
description: Create a structured handoff document for implementation sessions. Use when context is full after a long planning/discussion session, when handing off an implementation plan to a new session, or when the user says "handoff" in the context of an implementation task. Produces a handoff file that preserves discussion rationale and decisions, points to the plan file, and includes a continuation prompt for the next session.
---

# Implementation Handoff

Create a handoff document that preserves the discussion context — the decisions, rationale, and rejected alternatives — that led to an implementation plan. The plan file already exists separately; the handoff's job is to capture everything around it that would be lost when context resets.

## Step 1: Gather State

Run these commands and capture the output:
- `git branch --show-current`
- `git status`
- `git diff --stat`
- Locate the plan file (check `.claude/plans/` or ask the user)

## Step 2: Review Conversation

Scan the full conversation and extract:
- Every decision made, with the rationale and rejected alternatives
- Context that motivated the task — why now, what problem it solves
- Constraints or requirements that emerged during discussion
- Any caveats, unresolved questions, or honest uncertainty
- Parameter values, seeds, or specific details that matter
- Subtleties or gotchas discovered during discussion that the plan alone does not capture

## Step 3: Write Handoff File

Save to `.claude/handoffs/HANDOFF_YYYY-MM-DD_[slug].md` where slug is 3-5 descriptive words, lowercase, hyphen-separated.

Use this structure:

```
# Handoff: [Task Title]

**Branch**: [branch name]
**Date**: [today]
**Status**: [Plan approved / Partial implementation / Ready for implementation]
**Plan file**: [path to the plan file — MANDATORY]

---

## Motivation

[Why this work is being done. The problem it solves,
what prompted it, and the intended outcome.
1-2 paragraphs.]

## Decisions and Rationale

[The core of the handoff. For each significant decision:
- What was decided
- Why (the reasoning, not just the conclusion)
- What alternatives were considered and why they were rejected

This section should be detailed enough that someone reading it
months later understands not just WHAT was decided but WHY.
This is the information that the plan file does NOT contain.]

## Critical Subtleties

[Things discovered during discussion that are easy to get wrong
during implementation. Each item should explain the symptom
of getting it wrong, not just the rule. These may or may not
appear in the plan — the handoff captures them regardless.]

## Uncommitted Changes

[git status output, annotated with what each change does]

## Files to Read First

[Ordered list of files the next session should read,
with what to look for in each. Most important first.
The plan file should be in this list.]

## Caveats and Open Questions

[Honest list of things uncertain, unresolved, or potentially wrong.
This section is MANDATORY -- if genuinely no caveats, state
"None identified" and explain why confidence is high.]

---

## Continuation Prompt

[Ready-to-paste block for the next Claude session.
10-20 lines. Contains: task summary, pointer to BOTH this
handoff file and the plan file, key constraints,
"check git status and git branch before starting".]
```

## Step 4: Post-Handoff Checklist

- Update CLAUDE.md if project status changed
- Add a brief SESSION_LOG.md entry pointing to the handoff file (do not duplicate content):
  ```
  ## YYYY-MM-DD: [Brief Title]
  **Handoff**: `.claude/handoffs/HANDOFF_YYYY-MM-DD_slug.md`
  **Status**: Handed off for continuation
  ```
- Print the continuation prompt to the conversation so the user can copy it
- Remind the user about uncommitted changes if any exist

## Quality Rules

- No emojis
- All file paths relative to project root
- Exact numbers -- no rounding or paraphrasing metrics
- Caveats section is mandatory
- The plan file path is mandatory — the handoff complements the plan, it does not replace it
- The handoff must be readable by someone who was NOT in the session
- This handoff replaces the standard session wrap-up (working_guidelines.md Section 9, steps 1/2/4). Steps 3 and 5 still apply.
