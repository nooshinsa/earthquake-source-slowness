# GitHub Setup Notes

This repository is intended as a first public draft of the Python THETA
calculator converted from the original Fortran workflow.

## Recommended First Repository

Create a GitHub repository named something like:

```text
theta-python
```

For the first draft, a private repository is safest while validation is still
in progress. It can be made public later.

## Suggested Local Commands

From this folder:

```bash
git init
git add .gitignore GITHUB_SETUP.md python_code/*.py python_code/*.md python_code/*.txt python_code/*.csv
git commit -m "Initial Python THETA calculator draft"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/theta-python.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `theta-python` with your GitHub username and chosen
repository name.

## Files Intentionally Ignored

The `.gitignore` excludes downloaded waveforms, StationXML/RESP files, generated
results, and Python cache folders. Keep those locally unless you intentionally
want to publish a small validation dataset later.

## Next Packaging Step

After the scientific validation is stable, add:

```text
pyproject.toml
LICENSE
tests/
examples/
```

That can be done after the first GitHub draft.
