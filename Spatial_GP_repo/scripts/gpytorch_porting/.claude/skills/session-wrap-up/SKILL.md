---
name: session-wrap-up
description: Structured session wrap-up with documentation audit, hanging thread check, and git commit guidance. Use when the user says "wrap up", "done for now", "session end", or invokes /session-wrap-up. Supersedes the old wrap-up procedure in working_guidelines.md Section 9.
---

# Session Wrap-Up

Follow all steps in order.

## Step 1: Gather State

Run these commands:
```
git status
git branch --show-current
git diff --stat
```

Review the conversation to identify: what was accomplished, what files were modified, what decisions were made.

## Step 2: Summarize Accomplishments

Present 3-5 bullet points of what was done. Be specific (file names, function names, metric values).

## Step 3: Documentation Audit

Read `references/doc-hierarchy.md` for the full documentation map.

For each file modified this session, identify which documentation tier covers it:

1. **Check relevant rules and references** — Read the rule or reference file that covers the area. Check if it needs updating. Propose specific edits if needed.

2. **Check CLAUDE.md sections** — Verify:
   - Current Status table (component status changes?)
   - File Map (new files? deleted files?)
   - Parameter Matching Table (if parameters changed)
   - Known Issues (resolved? new?)
   - Deferred Items (completed? new?)

3. **Check for conflicting information** — If any doc contradicts what was done, flag it to the user. Do not silently resolve.

Present all proposed updates to the user for approval before making them.

## Step 4: Suggest New Documentation (if warranted)

If the session involved substantial work on a topic with NO existing rule or reference:

> "This session did significant work on [topic]. No dedicated doc exists for this area. Would you like to create a REFERENCE file (for solved/stable architecture) or a rule (for active gotchas that should auto-load)? Or is existing documentation sufficient?"

Only suggest when genuinely warranted. Most sessions do not need new files. See `references/doc-hierarchy.md` "When to Create New Documentation" for criteria.

## Step 5: Hanging Threads Check

Scan for loose ends:
- **DEBUG markers**: Grep for `DEBUG` in files modified this session
- **TODO comments**: Grep for `TODO` in git diff output
- **Investigation artifacts**: Check `investigations/` for uncleaned scripts
- **Deferred items**: Check if anything discussed but deferred should be added to CLAUDE.md

Report findings. Ask if DEBUG/TODO items should be cleaned up before committing.

## Step 6: SESSION_LOG.md Entry

Add a brief entry:
```markdown
## YYYY-MM-DD: [Brief Title]

**Branch**: `[branch name]`

**Accomplished:**
- [bullets from Step 2]

**Documentation updated:** [list files, or "none"]

**Known issues:** [new or unresolved, or "none"]
```

If a handoff was created, add: `**Handoff**: [path]`

## Step 7: Git Commit

1. Show `git status`, propose files to stage
2. Draft commit message (concise, "why" not "what")
3. Ask user for approval before committing
4. If the session involved a milestone change, remind user to run a canonical experiment
