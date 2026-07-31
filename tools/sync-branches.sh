#!/bin/sh
# Promote `testing` to `main`, keeping the two branches content-identical.
#
# pyCERR develops on `testing`; `main` should always hold the same tree. This
# script merges testing into main and refuses to push unless the resulting
# trees match exactly, so a partial or surprising merge cannot reach the
# public repository unnoticed.
#
# Nothing here runs on its own: this is a manual helper, invoked when you
# choose to promote. There is no CI job or hook that syncs the branches.
#
# Usage:
#   ./tools/sync-branches.sh          # merge locally and verify; does NOT push
#   ./tools/sync-branches.sh --push   # same, then push both branches
#
# Pushing is opt-in so a promotion can be inspected before it becomes public.
# It never force-pushes and never rewrites published history.

set -eu

DO_PUSH=0
case "${1:-}" in
    --push) DO_PUSH=1 ;;
    '') ;;
    *) echo "usage: $0 [--push]" >&2; exit 2 ;;
esac

SOURCE_BRANCH=testing
TARGET_BRANCH=main

die() { echo "error: $*" >&2; exit 1; }

# Refuse to run on a dirty tree: a merge would silently mix uncommitted work in.
if ! git diff --quiet || ! git diff --cached --quiet; then
    die "working tree has uncommitted changes; commit or stash them first"
fi

STARTING_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# Return to wherever the user was, even if the merge fails.
cleanup() { git checkout --quiet "$STARTING_BRANCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "Fetching..."
git fetch origin

# Work from the remote state so a stale local branch cannot silently drop
# commits another contributor has already pushed.
for BRANCH in "$SOURCE_BRANCH" "$TARGET_BRANCH"; do
    git checkout --quiet "$BRANCH"
    BEHIND=$(git rev-list --count "$BRANCH..origin/$BRANCH")
    AHEAD=$(git rev-list --count "origin/$BRANCH..$BRANCH")
    if [ "$BEHIND" -gt 0 ]; then
        echo "  $BRANCH is $BEHIND commit(s) behind origin; fast-forwarding"
        git merge --ff-only "origin/$BRANCH" \
            || die "$BRANCH has diverged from origin/$BRANCH; reconcile it manually"
    fi
    [ "$AHEAD" -gt 0 ] && echo "  $BRANCH is $AHEAD commit(s) ahead of origin (will be pushed)"
done

# Report what the promotion actually contributes before doing it.
echo
echo "Patches on $SOURCE_BRANCH not yet in $TARGET_BRANCH:"
git cherry "$TARGET_BRANCH" "$SOURCE_BRANCH" | grep '^+' \
    | while read -r _ SHA; do echo "  $(git log --oneline -1 "$SHA")"; done \
    || true
if git diff --quiet "$TARGET_BRANCH" "$SOURCE_BRANCH"; then
    echo "  (none - trees already identical)"
fi
echo
echo "Files that differ:"
git diff --stat "$TARGET_BRANCH" "$SOURCE_BRANCH" || echo "  (none)"

echo
echo "Merging $SOURCE_BRANCH into $TARGET_BRANCH..."
git checkout --quiet "$TARGET_BRANCH"
if git diff --quiet "$TARGET_BRANCH" "$SOURCE_BRANCH"; then
    echo "  Trees already identical; no merge commit needed."
else
    git merge --no-ff "$SOURCE_BRANCH" -m "Merge branch '$SOURCE_BRANCH' into $TARGET_BRANCH" \
        || die "merge conflict; resolve it, commit, then re-run"
fi

# The gate: promotion is only correct if the trees end up identical.
TARGET_TREE=$(git rev-parse "$TARGET_BRANCH^{tree}")
SOURCE_TREE=$(git rev-parse "$SOURCE_BRANCH^{tree}")
if [ "$TARGET_TREE" != "$SOURCE_TREE" ]; then
    die "trees differ after merge ($TARGET_TREE vs $SOURCE_TREE); refusing to push.
     Inspect with: git diff $TARGET_BRANCH $SOURCE_BRANCH"
fi
echo "  Trees match: $TARGET_TREE"

if [ "$DO_PUSH" -eq 0 ]; then
    echo
    echo "Merged locally and verified. Nothing pushed."
    echo "Review with: git log --oneline origin/$TARGET_BRANCH..$TARGET_BRANCH"
    echo "Then push with: $0 --push"
    exit 0
fi

echo
echo "Pushing..."
git push origin "$SOURCE_BRANCH"
git push origin "$TARGET_BRANCH"

echo
echo "Done. $SOURCE_BRANCH and $TARGET_BRANCH are aligned at tree $TARGET_TREE"
