#!/bin/bash
# Setup script for pre-commit hooks

echo "Setting up pre-commit hooks..."

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Install the git hook scripts
pre-commit install

# Install the pre-push hook
pre-commit install --hook-type pre-push

echo "✓ Pre-commit hooks installed!"
echo ""
echo "To run manually: pre-commit run --all-files"
echo "To update hooks: pre-commit autoupdate"
