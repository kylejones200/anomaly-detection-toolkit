# Documentation

This directory contains the Sphinx documentation for the Anomaly Detection Toolkit.

## Building Locally

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build the documentation
cd docs
make html

# View the documentation
# Open docs/_build/html/index.html in your browser
```

## Structure

- `conf.py`: Sphinx configuration
- `index.rst`: Main documentation page
- `getting_started.rst`: Quick start guide
- `user_guide/`: Detailed usage guides for each method type
- `api/`: Complete API reference (auto-generated)
- `examples/`: Code examples
- `contributing.rst`: Contribution guidelines

## Read the Docs

The documentation is automatically built and hosted on Read the Docs when:
- Code is pushed to the main branch
- A new release is tagged

Configuration is in `.readthedocs.yaml` at the repository root.
