# Building lunch-model

This document describes how to build the `lunch-model` package for deployment to another device.

## Prerequisites

- Python 3.12+
- `build` package: `pip install build`

## Build process

Run the build script:

```bash
./build.sh
```

This will:
1. Clean any previous builds
2. Build a Python wheel from `pyproject.toml`
3. Copy config files (`lite_llm_config.yaml`, `start.sh`, `kill.sh`, `readm.md`)
4. Generate an `install.sh` script for the target device
5. Create a tarball bundle

## Output

After build, the following files are created in `dist/`:

| File | Description |
|------|-------------|
| `lunch-model-X.X.X.tar.gz` | Deployable tarball (copy this to target) |
| `lunch_model-X.X.X-py3-none-any.whl` | Python wheel (inside tarball) |

## Inside the bundle

```
bundle/
├── lunch_model-X.X.X-py3-none-any.whl   # Python package
├── install.sh                             # Installs dependencies + package
├── start.sh                               # Starts all services
├── kill.sh                                # Stops all services
├── lite_llm_config.yaml                   # LiteLLM configuration
└── readm.md                               # Usage documentation
```

## Version

Version is read from `pyproject.toml` automatically:

```toml
[project]
version = "0.1.0"
```

## Entry points

The package exposes these commands when installed:

| Command | Description |
|---------|-------------|
| `qwen-server` | vLLM backend (port 11112) |
| `qwen-compress` | LiteLLM proxy (port 11111) |
| `qwen-stats` | Token stats server (port 11113) |

## Wheel-only build

To build just the wheel without the full bundle:

```bash
python3 -m build --wheel --outdir dist/
```

Then manually copy config files as needed.