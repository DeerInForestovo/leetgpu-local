#!/usr/bin/env bash
set -euo pipefail

# Generate the challenge branch from solution branch.
# Strips code between "// === BEGIN SOLUTION ===" and "// === END SOLUTION ==="
# markers and replaces it with a TODO comment.

SOLUTION_BRANCH="solution"
CHALLENGE_BRANCH="challenge"
PROBLEMS_DIR="problems"

# --- Sanity checks ---

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$SOLUTION_BRANCH" ]]; then
    echo "Error: must run from the '$SOLUTION_BRANCH' branch (currently on '$current_branch')"
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is dirty. Commit or stash changes first."
    exit 1
fi

# --- Strip solutions into a temp dir ---

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cp -r "$PROBLEMS_DIR" "$tmp/$PROBLEMS_DIR"

for f in "$tmp/$PROBLEMS_DIR"/*.cu; do
    [ -f "$f" ] || continue
    # Replace everything between BEGIN/END markers with a TODO placeholder.
    # Handles multiple marker pairs per file.
    awk '
    /\/\/ === BEGIN SOLUTION ===/ {
        inside = 1
        print "        // TODO: implement your GPU solution here"
        next
    }
    /\/\/ === END SOLUTION ===/ {
        inside = 0
        next
    }
    !inside { print }
    ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

# --- Apply to challenge branch ---

solution_head=$(git rev-parse --short HEAD)

git checkout "$CHALLENGE_BRANCH"
cp -f "$tmp/$PROBLEMS_DIR"/*.cu "$PROBLEMS_DIR/"
git add "$PROBLEMS_DIR/"

if git diff --cached --quiet; then
    echo "No changes to commit. Challenge branch is already up to date."
else
    git commit -m "sync from $SOLUTION_BRANCH ($solution_head): strip solutions"
    echo "Challenge branch updated."
fi

git checkout "$SOLUTION_BRANCH"
echo "Back on $SOLUTION_BRANCH. Done."
