# Contributing

## Development setup

```bash
make install   # create .venv and install all dev + doc dependencies
make check     # lint + typecheck + test (full CI gate)
```

## Workflow

This project follows strict TDD. Before writing any implementation code:

1. Pick a task from `bd ready`
2. Write a failing test in `tests/` that captures the acceptance criteria
3. Run `make test` — confirm it fails for the right reason
4. Write the minimum implementation to make it pass
5. Run `make check` before closing the task

Never write implementation code without a corresponding failing test driving it.

## Releasing a new version

Releases are fully automated once a version tag is pushed.

**Steps:**

1. Bump `version` in `pyproject.toml`
2. Commit the bump:
   ```bash
   git commit -am "chore: bump to vX.Y.Z"
   git push
   ```
3. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The [`publish` workflow](https://github.com/heliopais/interlace/actions/workflows/publish.yml)
triggers automatically on `v*` tags and:

- Runs the full CI gate (lint + typecheck + tests)
- Publishes the wheel and sdist to [PyPI](https://pypi.org/project/interlace-lme/)
  (silently skips if the version already exists)
- Creates a [GitHub Release](https://github.com/heliopais/interlace/releases)
  with auto-generated release notes

> **Note:** Do not push a tag without first bumping the version in `pyproject.toml` —
> PyPI will reject a duplicate version and the workflow will skip the publish step.
