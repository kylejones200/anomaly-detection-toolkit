# PyPI Trusted Publisher Setup

This guide explains how to set up trusted publishing to PyPI using GitHub Actions OIDC.

## Prerequisites

1. PyPI account (create at https://pypi.org/account/register/)
2. GitHub repository with Actions enabled
3. Package project configured in `pyproject.toml`

## Step 1: Configure Trusted Publisher on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Scroll down to **"Add a new pending publisher"**
3. Fill in the form:
   - **PyPI project name**: `anomaly-detection-toolkit`
   - **Owner**: `kylejones200`
   - **Repository name**: `anomaly-detection-toolkit`
   - **Workflow filename**: `publish.yml`
   - **Environment name**: (leave empty or set to `pypi` if you want to use an environment)

4. Click **"Add pending publisher"**

5. **Important**: The trusted publisher must match the **subject** (sub) claim from the OIDC token. Based on the error message, you should use:
   - **Subject**: `repo:kylejones200/anomaly-detection-toolkit:ref:refs/tags/v*`

   OR for broader support:
   - **Subject**: `repo:kylejones200/anomaly-detection-toolkit:ref:refs/heads/main`

## Step 2: Update PyPI Trusted Publisher Configuration

After adding the pending publisher, you'll need to configure it with the correct subject pattern. PyPI's UI may require you to specify the exact pattern.

**Recommended Subject Patterns:**

For tag-based publishing:
```
repo:kylejones200/anomaly-detection-toolkit:ref:refs/tags/v*
```

For main branch publishing (alternative):
```
repo:kylejones200/anomaly-detection-toolkit:ref:refs/heads/main
```

For any branch/tag:
```
repo:kylejones200/anomaly-detection-toolkit:*
```

## Step 3: Verify Workflow Configuration

The workflow file `.github/workflows/publish.yml` is already configured correctly with:
- `permissions.id-token: write` (required for OIDC)
- Using `pypa/gh-action-pypi-publish@release/v1` action

## Step 4: Test the Publishing

Once the trusted publisher is configured:

1. Push a new tag:
   ```bash
   git tag -a v0.1.1 -m "Test release"
   git push origin v0.1.1
   ```

2. Or create a GitHub Release which will also trigger the workflow

3. Check the GitHub Actions tab to see if the workflow runs successfully

## Troubleshooting

### Error: "invalid-publisher"

This means the trusted publisher configuration doesn't match the OIDC token claims. Check:

1. **Repository name** matches exactly: `kylejones200/anomaly-detection-toolkit`
2. **Workflow filename** matches: `publish.yml`
3. **Subject pattern** matches the tag/branch pattern you're using

### Check OIDC Token Claims

The workflow will show the token claims in the error message. Use those exact values to configure the trusted publisher.

From the error, the claims are:
- `sub`: `repo:kylejones200/anomaly-detection-toolkit:ref:refs/tags/v0.1.0`
- `repository`: `kylejones200/anomaly-detection-toolkit`
- `workflow_ref`: `kylejones200/anomaly-detection-toolkit/.github/workflows/publish.yml@refs/tags/v0.1.0`

### Using Environment-based Publishing

If you want more control, you can use GitHub Environments:

1. Go to repository Settings → Environments
2. Create a new environment called `pypi`
3. Add a protection rule if needed
4. Update the workflow to reference the environment:
   ```yaml
   environment: pypi
   ```

Then use this subject pattern:
```
repo:kylejones200/anomaly-detection-toolkit:environment:pypi
```

## Alternative: Using API Tokens

If OIDC setup is problematic, you can use API tokens (less secure but simpler):

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Add it as a GitHub Secret: `PYPI_API_TOKEN`
4. Update the workflow to use the token instead of OIDC

However, OIDC is recommended for better security.

## References

- [PyPI Trusted Publishers Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [Troubleshooting Trusted Publishers](https://docs.pypi.org/trusted-publishers/troubleshooting/)
