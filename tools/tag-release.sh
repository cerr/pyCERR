#!/bin/sh
# Tag a release on the commit shared by `testing` and `main`.
#
# setuptools_scm derives the package version from the nearest tag reachable
# from the checked-out commit. Promotion is a fast-forward
# (tools/sync-branches.sh), so after it both branches end on the same commit;
# tagging that commit gives the exact release version on `testing`, on `main`
# and in the PyPI build.
#
# Run this after the release commit is on `testing` and has been promoted with
# ./tools/sync-branches.sh --push.
#
# Usage:
#   ./tools/tag-release.sh v2.3.3          # create the tag locally; does NOT push
#   ./tools/tag-release.sh v2.3.3 --push   # same, then push the tag
#
# Pushing the tag triggers the PyPI publish workflow (.github/workflows/publish.yml),
# so it is opt-in. Existing tags are never moved or overwritten.

set -eu

TAG="${1:-}"
DO_PUSH=0
case "${2:-}" in
    --push) DO_PUSH=1 ;;
    '') ;;
    *) echo "usage: $0 vX.Y.Z [--push]" >&2; exit 2 ;;
esac

SOURCE_BRANCH=testing
TARGET_BRANCH=main

die() { echo "error: $*" >&2; exit 1; }

echo "$TAG" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$' \
    || { echo "usage: $0 vX.Y.Z [--push]" >&2; exit 2; }
VERSION=${TAG#v}

echo "Fetching..."
git fetch origin --tags

if git rev-parse --quiet --verify "refs/tags/$TAG" >/dev/null; then
    die "tag $TAG already exists; published tags are never moved"
fi

# Tag the published tip of the branches, not whatever happens to be checked out.
RELEASE_COMMIT=$(git rev-parse "origin/$SOURCE_BRANCH")
LOCAL_COMMIT=$(git rev-parse "$SOURCE_BRANCH")
if [ "$RELEASE_COMMIT" != "$LOCAL_COMMIT" ]; then
    die "$SOURCE_BRANCH and origin/$SOURCE_BRANCH differ; push or pull first"
fi

# The release must already be promoted: main is on the same commit as testing.
if [ "$(git rev-parse "origin/$TARGET_BRANCH")" != "$RELEASE_COMMIT" ]; then
    die "origin/$TARGET_BRANCH is not at origin/$SOURCE_BRANCH;
     promote first with ./tools/sync-branches.sh --push"
fi

# The changelog at the release commit must describe this version.
if ! git show "$RELEASE_COMMIT:CHANGELOG.md" | grep -q "^## \[$VERSION\]"; then
    die "CHANGELOG.md at $SOURCE_BRANCH has no '## [$VERSION]' section"
fi

git tag -a "$TAG" -m "$TAG" "$RELEASE_COMMIT"
echo "Tagged $(git log --oneline -1 "$RELEASE_COMMIT") as $TAG"
echo "Version seen from $SOURCE_BRANCH: $(git describe --tags "$SOURCE_BRANCH")"
echo "Version seen from $TARGET_BRANCH:    $(git describe --tags "origin/$TARGET_BRANCH")"

if [ "$DO_PUSH" -eq 0 ]; then
    echo
    echo "Tag created locally. Nothing pushed."
    echo "Inspect with: git show $TAG"
    echo "Publish with: git push origin $TAG"
    exit 0
fi

echo
echo "Pushing $TAG (this triggers the PyPI publish workflow)..."
git push origin "$TAG"
echo "Done."
