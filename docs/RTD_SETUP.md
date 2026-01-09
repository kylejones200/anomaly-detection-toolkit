# Read the Docs Setup Guide

This guide explains how to set up your project on Read the Docs.

## Prerequisites

1. Have your project on GitHub (public or private)
2. Have a Read the Docs account (https://readthedocs.org/)

## Setup Steps

### 1. Import Project on Read the Docs

1. Go to https://readthedocs.org/dashboard/
2. Click "Import a Project"
3. Select "Import manually"
4. Enter project details:
   - **Name**: `anomaly-detection-toolkit`
   - **Repository URL**: `https://github.com/kylejones200/anomaly-detection-toolkit`
   - **Repository type**: Git
5. Click "Next"

### 2. Configure Project Settings

In the project settings on Read the Docs:

1. **Basic Settings**:
   - **Name**: `anomaly-detection-toolkit`
   - **Language**: English
   - **Programming Language**: Python
   - **Project URL**: (auto-generated)
   - **Repository**: (auto-filled)

2. **Advanced Settings**:
   - **Python configuration file**: `.readthedocs.yaml`
   - **Install Project**: Yes
   - **Python version**: 3.12
   - **Use system packages**: No (recommended)

3. **Build Settings**:
   - **Documentation type**: Sphinx
   - **Sphinx configuration file**: `docs/conf.py`
   - **Requirements file**: `docs/requirements.txt`

### 3. Verify Configuration

The `.readthedocs.yaml` file in the root directory is already configured:

- Python 3.12
- Sphinx with `docs/conf.py`
- Installs package with `docs` extra requirements

### 4. Build Documentation

After importing, Read the Docs will automatically:
1. Clone your repository
2. Install dependencies (including `docs` extras)
3. Build the documentation
4. Host it at: `https://anomaly-detection-toolkit.readthedocs.io/`

### 5. Automatic Builds

Read the Docs will automatically rebuild documentation when:
- You push to the `main` branch
- You create a new release/tag
- You manually trigger a build from the dashboard

### 6. Version Management

By default, Read the Docs builds:
- **Latest** (from main branch)
- **Stable** (from latest tag)
- You can add more versions in Settings > Versions

## Troubleshooting

### Build Failures

Check the build logs in the Read the Docs dashboard:
1. Go to your project dashboard
2. Click "Builds"
3. Click on the failed build to see logs

Common issues:
- Missing dependencies: Add to `docs/requirements.txt` or `pyproject.toml` `[project.optional-dependencies]` `docs`
- Import errors: Check `autodoc_mock_imports` in `docs/conf.py`
- Configuration errors: Verify `.readthedocs.yaml` syntax

### Missing Modules

If autodoc can't find modules, ensure:
- Package is properly installed: `pip install -e .`
- Python path is correct in `conf.py`
- `sys.path.insert` is set if needed

### Theme Issues

The documentation uses `sphinx-rtd-theme`. Ensure it's in `docs/requirements.txt`.

## Testing Locally

Before pushing, test the documentation build locally:

```bash
pip install -e ".[docs]"
cd docs
make html
# Check output in docs/_build/html/
```

## Custom Domain (Optional)

You can set up a custom domain:
1. Go to Settings > Domains
2. Add your custom domain
3. Follow DNS configuration instructions
