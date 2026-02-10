#!/bin/bash
# Hook: worktree-check.sh
# Fires on SessionStart (startup, resume, compact, clear).
# Checks git worktree status and injects context so Claude presents
# a concise branch/worktree summary to the user.

BRANCH=$(git branch --show-current 2>/dev/null || echo "UNKNOWN")
WORKTREES=$(git worktree list 2>/dev/null || echo "ERROR: git worktree list failed")
NUM_WORKTREES=$(echo "$WORKTREES" | wc -l)

# Build the context message
MSG="GIT WORKTREE CHECK (auto-injected by SessionStart hook):
Current branch: $BRANCH
Working directory: $(pwd)
Worktrees ($NUM_WORKTREES total):
$WORKTREES

INSTRUCTIONS FOR CLAUDE:
At the start of the conversation, give a concise one-line summary of branch and worktree status.
Example: 'Branch: pietro/acquisition-functions (worktree: gpytorch_porting_acquisition, 2 worktrees active)'

If on pietro/workingbranch: this is the default working branch — no warning needed, skip the summary.

If on a feature/investigation branch: mention it briefly so the user can confirm it matches their intent.

CRITICAL — If multiple worktrees are active: You MUST tell the user immediately, even before any other work.
Report how many worktrees exist, which directory and branch each one is on, and which one THIS session is in.
This applies regardless of which branch we are on (including pietro/workingbranch). Never skip this.
Flag if the branch name seems mismatched with the session context (e.g., on arcsine-kernel but discussing acquisition functions).

DO NOT use 'git checkout' or 'git switch' to change branches when multiple worktrees exist.
Guide the user through 'git worktree add' instead."

jq -n --arg msg "$MSG" '{"additionalContext": $msg}'
