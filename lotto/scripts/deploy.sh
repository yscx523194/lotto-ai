#!/bin/bash
# Lotto Auto Purchase - Deployment Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Lotto - Deployment"
echo "========================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# Update Environment (Dependencies & Playwright)
echo "Updating environment..."
"$SCRIPT_DIR/setup-env.sh"
echo ""

# Update Systemd Timer
if command -v systemctl &> /dev/null; then
    echo "Updating systemd timer..."
    "$SCRIPT_DIR/install-systemd.sh"
else
    echo "Systemd not found. Skipping timer update."
fi

echo ""
echo "Deployment completed successfully!"

