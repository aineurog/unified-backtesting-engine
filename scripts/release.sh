#!/usr/bin/env bash
# Build and upload the current version to PyPI using ~/.pypirc (or TWINE_*).
#
# Usage: scripts/release.sh [--yes] [--dry-run]
#   --yes      skip the confirmation prompt
#   --dry-run  build and check the distributions, but do not upload
#
# The working tree must be clean and the two version strings must agree. After a
# successful upload, tag the commit and push the tag if you want a GitHub release:
#   git tag -a vX.Y.Z -m "UBE X.Y.Z" && git push origin vX.Y.Z
set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
cd "$repo_root"

confirm=0
dry_run=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) confirm=1 ;;
        --dry-run) dry_run=1; confirm=1 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$arg'" >&2; exit 2 ;;
    esac
done

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is not clean; commit or stash first" >&2
    exit 1
fi

pyproject_version=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
module_version=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/ube/__init__.py)
if [ "$pyproject_version" != "$module_version" ]; then
    echo "error: pyproject version ($pyproject_version) != ube.__version__ ($module_version)" >&2
    exit 1
fi

if [ "$confirm" -ne 1 ]; then
    read -r -p "Build and upload unified-backtesting-engine $pyproject_version to PyPI? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "aborted"; exit 1 ;;
    esac
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install -q --upgrade build twine

rm -rf dist
"$work/venv/bin/python" -m build
"$work/venv/bin/python" -m twine check dist/*
if [ "$dry_run" -eq 1 ]; then
    echo "dry run: built $(ls dist/* | tr '\n' ' ')—skipping upload"
else
    "$work/venv/bin/python" -m twine upload --non-interactive dist/*
fi
