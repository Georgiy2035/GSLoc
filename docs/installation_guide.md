# Installation Guide

This guide provides step-by-step instructions for installing and configuring the multimodal place recognition system. Before proceeding, verify that your environment meets the [platform requirements](platform_requirements.md). Two installation methods are supported:

- [**Local with uv**](#local-development-with-uv) – Fast iteration, native performance, recommended for active development
- [**Docker**](#docker-installation) – Isolated environment, reproducible builds, recommended for deployment and CI/CD

## Local Development with uv

This method uses `uv` to manage Python dependencies and virtual environments, providing fast package resolution and installation for local development workflows.

### Prerequisites: Installing uv

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package and project manager written in Rust by Astral. It provides a unified interface that replaces multiple tools (`pip`, `virtualenv`, `pyenv`, etc.) with 10-100× faster performance.

**Official resources:**
- [Documentation](https://docs.astral.sh/uv/)
- [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- [GitHub repository](https://github.com/astral-sh/uv)

**Installation:**

On Linux/macOS, install via the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, install via pip:

```bash
pip install uv
```

Verify installation:

```bash
uv --version
```

### Environment Setup

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd multimodal-place-recognition
```

Initialize the uv environment and install dependencies:

```bash
# Install core dependencies + dev group (linting tools)
uv sync

# Or install full development environment (includes PyTorch, notebooks, visualization)
uv sync --group dev-full
```

This will:
1. Create a virtual environment in `.venv/`
2. Install the project in editable mode
3. Install all dependencies according to the locked `uv.lock` file

### Understanding Dependency Groups

The project uses dependency groups to organize optional dependencies by purpose. This allows you to install only what you need for your workflow.

#### Available Groups

| Group | Contents | Use Case |
|-------|----------|----------|
| **`dev`** | Linting tools (ruff, pre-commit) | Basic development (installed by default) |
| **`dev-full`** | All development tools | Complete development setup |
| **`torch`** | PyTorch 2.7.1 + CUDA 12.6 | Deep learning (pinned versions) |
| **`faiss-cpu`** | Faiss-CPU 1.13.1 | FAISS (pinned versions) |
| **`lint`** | Code quality tools | CI/CD linting jobs |
| **`notebook`** | Jupyter, IPython widgets | Interactive development |
| **`viz`** | Rerun, matplotlib, plotly | Visualization and debugging |

#### Installation Commands

**Minimal setup** (core dependencies + linting only):
```bash
uv sync
```

**Full development environment** (recommended for most developers):
```bash
uv sync --group dev-full
```

**Custom combinations** (install specific groups):
```bash
# Core + PyTorch only
uv sync --group torch

# Core + notebooks + visualization (without PyTorch)
uv sync --group notebook --group viz

# Core + all groups
uv sync --all-groups
```

**CI/CD usage** (install only linting tools, skip project installation):
```bash
uv sync --only-group lint
```

#### PyTorch Installation Notes

⚠️ **Important:** PyTorch is intentionally excluded from the core dependencies to allow flexibility in deployment environments. The `torch` dependency group installs PyTorch 2.7.1 with CUDA 12.6 support for development.

- For **development**: Use `uv sync --group torch` or `uv sync --group dev-full`
- For **deployment**: Install PyTorch manually based on your target hardware (CPU, different CUDA versions, etc.)
- The project requires PyTorch ≥2.4.1 (defined in [platform requirements](platform_requirements.md))

### Activating the Environment

After installation, activate the virtual environment:

```bash
# Automatic activation with uv run
uv run python your_script.py

# Or activate manually
source .venv/bin/activate  # Linux/macOS
```

### Updating Dependencies

To upgrade locked dependencies to newer versions:

```bash
# Upgrade all packages
uv sync --upgrade

# Upgrade specific package
uv sync --upgrade-package torch
```

## Docker Installation

This method builds a containerized environment with all dependencies pre-configured, ensuring consistency across different systems and simplifying deployment.

_work in progress..._
