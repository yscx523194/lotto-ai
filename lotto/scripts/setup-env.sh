#!/bin/bash
# Lotto Auto Purchase - Environment Setup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Lotto - Environment Setup"
echo "========================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# Check Python
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
else
    echo "Virtual environment already exists"
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
echo "pip upgraded"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
echo "Dependencies installed"
echo ""

# Install Playwright browsers
echo "Installing Playwright browsers..."
"$VENV_DIR/bin/playwright" install chromium

# Install system dependencies on Linux (requires sudo)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v pacman &> /dev/null; then
        echo "Arch Linux (or derivative) detected. Installing minimal headless dependencies..."
        echo "Sudo password may be required."
        # User requested minimal list (Apt -> Arch mapping)
        sudo pacman -S --needed --noconfirm \
            nss nspr at-spi2-core libxkbcommon libxcomposite \
            libxdamage libxrandr mesa alsa-lib pango cairo gdk-pixbuf2 \
            tesseract
    else
        echo "Linux detected: Installing system dependencies for headless browser..."
        echo "Sudo password may be required."
        sudo "$VENV_DIR/bin/playwright" install-deps chromium || echo "Warning: playwright install-deps failed. Continuing anyway..."
        
        if command -v apt-get &> /dev/null; then
            echo "Installing tesseract-ocr for Ubuntu/Debian..."
            sudo apt-get update && sudo apt-get install -y tesseract-ocr
        fi
    fi
fi

echo "Playwright Chromium browser installed"
echo ""

# Check .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "Creating .env from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ".env file created"
    echo "Please edit .env and configure your settings"
else
    echo ".env file exists"
fi
echo ""

echo "Environment setup completed!"
echo ""
echo "Useful commands:"
echo "  • Configure .env:       nano .env"
echo "  • Run manually:         ./scripts/run.sh"
echo "  • Install systemd:      ./scripts/install-systemd.sh"
echo ""
