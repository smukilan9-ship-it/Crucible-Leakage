# Releasing

Four things have to be true before a version goes out, and three of them are
checkable by running something.

## Before you upload

**Two fields in `CITATION.cff` are still open: `family-names` and `orcid`.**
Neither can be filled from the source tree, and a citation missing them is
harder for a reader to follow, so add them before the Zenodo deposit. The
repository URLs in `pyproject.toml` and `CITATION.cff` are already set.

**Bump the version in two places, or the interface goes stale.** `pyproject.toml`
holds the real version; `web/static/index.html` pins `app.js` and `style.css` to
it with a query string, which is what makes a redeploy a new URL for a browser
that has already cached the old one. A test fails if the two drift apart, so you
will find out, but it is quicker to remember.

## The checks

```bash
pip install -e ".[dev]"
pytest                                   # 66 tests, none of which touch a network
python -m build                          # builds the wheel and the sdist
python -m twine check dist/*             # PyPI's own metadata validation
```

Then install the built artifact somewhere clean and confirm the entry point
exists, because a broken `console_scripts` entry is invisible until someone
installs it:

```bash
python -m venv /tmp/verify && /tmp/verify/bin/pip install dist/*.whl
/tmp/verify/bin/crucible models
```

## Uploading

```bash
python -m twine upload dist/*
```

Test PyPI first if this is the first release of a version number, because a
version can never be reused once uploaded, even after deletion:

```bash
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ crucible-leakage
```

## After

**Tag the commit** with the same version. A DOI minted from an untagged tree
points at something that will move.

**Deposit the tagged release**, which is what turns the software into something
citable. The `CITATION.cff` at the tag is what an archive reads, so it has to be
correct before the tag rather than after.

## Names

`crucible-leakage` is the distribution; `crucible` is the import name and the
command. Plain `crucible` is already taken on PyPI, which is why the
distribution carries the longer name.
