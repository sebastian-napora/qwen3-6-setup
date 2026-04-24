#!/bin/bash
#
# Build script for lunch-model package
# Packages the server code for deployment to another device
#
# Usage:
#   ./build.sh              # build wheel + bundle
#   ./build.sh wheel-only   # build wheel only (no bundle)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Version from pyproject.toml (sed to avoid python import)
VERSION=$(grep -m1 '^version = ' pyproject.toml | sed 's/.*"\(.*\)"/\1/')
PKG_NAME=$(grep -m1 '^name = ' pyproject.toml | sed 's/.*"\(.*\)"/\1/')

echo "========================================"
echo "  lunch-model build script v$VERSION"
echo "========================================"

# Clean old builds
echo ""
echo "[1/4] Cleaning old builds..."
rm -rf build/ dist/ *.egg-info .pytest_cache
rm -f "$PKG_NAME-$VERSION-py3-none-any.whl"

# Create dist directory
mkdir -p dist/bundle

# Build Python wheel
echo ""
echo "[2/4] Building Python wheel..."
python3 -m pip install --quiet build
python3 -m build --wheel --outdir dist/

WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: Wheel build failed!"
    exit 1
fi
echo "   Built: $WHEEL_FILE"

# Copy config files into dist/bundle/
echo ""
echo "[3/4] Copying config files..."
cp lite_llm_config.yaml dist/bundle/ 2>/dev/null || true
cp start.sh dist/bundle/ 2>/dev/null || true
cp kill.sh dist/bundle/ 2>/dev/null || true
cp readm.md dist/bundle/ 2>/dev/null || true
cp build.md dist/bundle/ 2>/dev/null || true
cp run-build.md dist/bundle/ 2>/dev/null || true
mkdir -p dist/bundle/scripts

# Create install.sh (will be copied to bundle)
cat > dist/bundle/install.sh << 'INSTALL_EOF'
#!/bin/bash
#
# Install script for lunch-model
# Run this on the target device after copying the bundle
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  lunch-model install script"
echo "========================================"

# Check Python version
PYTHON_MAJOR=$(python3 --version 2>&1 | sed 's/Python //' | cut -d. -f1)
if [ "$PYTHON_MAJOR" -lt 3 ]; then
    echo "ERROR: Python 3.12+ required, found $PYTHON_MAJOR"
    exit 1
fi
echo "   Python 3 found"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "   (GPU info unavailable)"
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers + CUDA are installed."
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

VENV_PYTHON="venv/bin/python3"

# Upgrade pip
echo ""
echo "Upgrading pip..."
$VENV_PYTHON -m pip install --upgrade pip

# Install wheel
WHEEL_FILE=$(ls *.whl 2>/dev/null | head -1)
if [ -n "$WHEEL_FILE" ]; then
    echo ""
    echo "Installing lunch-model package..."
    $VENV_PYTHON -m pip install "$WHEEL_FILE"
else
    echo "WARNING: No wheel file found. Install manually: pip install *.whl"
fi

# Install runtime dependencies
echo ""
echo "Installing runtime dependencies..."
$VENV_PYTHON -m pip install vllm litellm fastapi uvicorn aiohttp

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit lite_llm_config.yaml if needed"
echo "  2. Run: ./start.sh"
echo ""
INSTALL_EOF
chmod +x dist/bundle/install.sh

# Copy wheel to bundle (MUST happen before tarball creation)
cp "$WHEEL_FILE" dist/bundle/

# Create tarball bundle
echo ""
echo "[4/4] Creating deployable tarball..."
tar -czf "dist/lunch-model-$VERSION.tar.gz" \
    -C dist bundle/

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "   dist/$WHEEL_FILE"
echo "   dist/lunch-model-$VERSION.tar.gz"
echo ""
echo "To deploy:"
echo "   1. Copy tarball to target device"
echo "   2. Extract: tar -xzf lunch-model-$VERSION.tar.gz"
echo "   3. Run: cd bundle && ./install.sh"
echo "   4. Run: ./start.sh"
echo ""