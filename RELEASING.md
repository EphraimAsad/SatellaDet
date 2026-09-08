# Releasing SatellaDet

This repository is designed to be published from a **fresh public Git repository**, without importing private development history.

## 1. Confirm release rights

Before the first public release, confirm that the framework source code can be released under Apache-2.0 and that no employer, client, collaborator, or third-party agreement prevents publication.

Do not use this source-code release as evidence that any dataset or trained weights are cleared for publication. Review those separately.

## 2. Run the local release checks

```bash
python tools/privacy_scan.py .
python -m pytest
```

The privacy scanner is heuristic. Manually inspect the final archive as well.

## 3. Build distributions

Install the development extras and build frontend:

```bash
python -m pip install -e ".[dev]"
python -m build
```

A normal pure-Python release should produce a source distribution and one `py3-none-any` wheel.

## 4. Test installation

Create a clean environment and install the built wheel:

```bash
python -m venv .release-test
# activate the environment, then:
python -m pip install dist/satelladet-0.2.1-py3-none-any.whl
satelladet-train --help
```

## 5. Create the public repository

Copy this sanitized source tree into an empty directory and initialize a fresh repository:

```bash
git init
git add .
git commit -m "Initial SatellaDet open-source release"
```

Do not copy private `.git` history into the public repository.

## 6. Publish to PyPI

The recommended automated route is PyPI Trusted Publishing through the included GitHub Actions `release.yml` workflow. Configure the repository/workflow as a Trusted Publisher in PyPI before using it.

Trusted Publishing avoids storing a long-lived PyPI API token in the repository or GitHub secrets.

Before the first publication, verify that the `satelladet` distribution name is still available on PyPI. If it is not, change only the distribution name in `pyproject.toml`; the Python import package can remain `satelladet`.

## 7. Tag the release

After tests pass and the release contents are reviewed:

```bash
git tag v0.2.1
git push origin v0.2.1
```

The included release workflow is configured to build and publish tags beginning with `v`.
