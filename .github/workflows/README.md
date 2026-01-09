# GitHub Actions Workflows

## PyPI Publishing Workflow

The `publish.yml` workflow automatically publishes your package to PyPI when:
- A GitHub release is published
- Manually triggered via workflow dispatch

### Setup Requirements

Since you're using OIDC (OpenID Connect) for PyPI:

1. **Configure Trusted Publisher in PyPI:**
   - Go to your PyPI project settings: https://pypi.org/manage/project/{project-name}/publishing/
   - Add a trusted publisher
   - Set:
     - **Publisher**: `GitHub Actions`
     - **Owner**: `kylejones200` (your GitHub username/org)
     - **Repository**: `kylejones200/anomaly-detection-toolkit`
     - **Workflow filename**: `.github/workflows/publish.yml`
     - **Environment name**: (leave empty or specify if using environments)

2. **The workflow will:**
   - Build the package using `python -m build`
   - Check the package with `twine check`
   - Publish to PyPI using OIDC authentication (no tokens needed!)

### Manual Publishing

To manually trigger a release:
1. Go to Actions tab in GitHub
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Optionally specify a version

### Automated Publishing

To automatically publish on releases:
1. Create a new release in GitHub
2. Tag with version (e.g., `v0.1.0`)
3. Publish the release
4. The workflow will automatically build and publish

## Test Workflow

The `test.yml` workflow runs tests on:
- Push to main/develop branches
- Pull requests to main/develop
- Manual trigger

Tests run on Python 3.12 and 3.13.
