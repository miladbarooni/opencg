# GitHub Actions Workflows

This directory contains CI/CD workflows for OpenCG.

## Workflows

### `tests.yml` - Test Suite
Runs on every push and PR to `main` and `develop` branches.

**Jobs**:
1. **test**: Runs unit and integration tests
   - Python versions: 3.9, 3.10, 3.11
   - Operating systems: Ubuntu, macOS
   - Verifies C++ backend compilation
   - Generates coverage report

2. **lint**: Code quality checks
   - Ruff (fast Python linter)
   - Black (code formatting)
   - isort (import sorting)

3. **build-wheels**: Build distribution wheels
   - Creates wheel packages for all platforms
   - Uploads as artifacts

## Badges

Add these badges to your README.md:

```markdown
![Tests](https://github.com/miladbarooni/opencg/actions/workflows/tests.yml/badge.svg)
![Coverage](https://codecov.io/gh/miladbarooni/opencg/branch/main/graph/badge.svg)
```

## Local Testing

Before pushing, run tests locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linters
ruff check opencg/
black --check opencg/ tests/
isort --check-only opencg/ tests/

# Auto-fix formatting
black opencg/ tests/
isort opencg/ tests/
```

## Coverage

Coverage reports are automatically uploaded to Codecov. You'll need to:
1. Sign up at https://codecov.io
2. Enable your repository
3. No token needed for public repos
