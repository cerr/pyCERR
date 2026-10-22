#!/bin/sh
# Promote `testing` to `main` by fast-forward, so both branches end on the
# same commit.
#
# pyCERR develops on `testing`; `main` is fast-forwarded to it. Because the two
# branches then share every commit, a release tag is reachable from both and
# setuptools_scm reports the same version on either (see tools/tag-release.sh).
# The script refuses to promote if `main` holds commits that `testing` lacks,
# and refuses to push unless both branches end on the same commit.
#
# Nothing here runs on its own: this is a manual helper, invoked when you
# choose to promote. There is no CI job or hook that syncs the branches.
#
# Usage:
#   ./tools/sync-branches.sh          # fast-forward locally and verify; does NOT push
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

# Refuse to run on a dirty tree: switching branches would carry uncommitted work along.
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

# main must be an ancestor of testing, otherwise promotion is not a
# fast-forward. That only happens if something was committed straight to main.
if ! git merge-base --is-ancestor "$TARGET_BRANCH" "$SOURCE_BRANCH"; then
    echo
    echo "Commits on $TARGET_BRANCH that are not in $SOURCE_BRANCH:"
    git log --oneline "$SOURCE_BRANCH..$TARGET_BRANCH"
    die "$TARGET_BRANCH has commits that $SOURCE_BRANCH lacks; cannot fast-forward.
     Bring them into $SOURCE_BRANCH first (git checkout $SOURCE_BRANCH && git merge $TARGET_BRANCH),
     then re-run."
fi

echo
echo "Fast-forwarding $TARGET_BRANCH to $SOURCE_BRANCH..."
git checkout --quiet "$TARGET_BRANCH"
git merge --ff-only "$SOURCE_BRANCH" || die "fast-forward failed"

# The gate: promotion is only correct if both branches end on the same commit.
TARGET_COMMIT=$(git rev-parse "$TARGET_BRANCH")
SOURCE_COMMIT=$(git rev-parse "$SOURCE_BRANCH")
if [ "$TARGET_COMMIT" != "$SOURCE_COMMIT" ]; then
    die "branches differ after fast-forward ($TARGET_COMMIT vs $SOURCE_COMMIT); refusing to push."
fi
echo "  Both branches at: $(git log --oneline -1 "$TARGET_COMMIT")"

if [ "$DO_PUSH" -eq 0 ]; then
    echo
    echo "Fast-forwarded locally and verified. Nothing pushed."
    echo "Review with: git log --oneline origin/$TARGET_BRANCH..$TARGET_BRANCH"
    echo "Then push with: $0 --push"
    exit 0
fi

echo
echo "Pushing..."
git push origin "$SOURCE_BRANCH"
git push origin "$TARGET_BRANCH"

echo
echo "Done. $SOURCE_BRANCH and $TARGET_BRANCH are aligned at $TARGET_COMMIT"
echo "To release this commit: ./tools/tag-release.sh vX.Y.Z"
