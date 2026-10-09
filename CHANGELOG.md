# Changelog

Notable project changes are recorded here. This file summarizes repository changes; it does not imply that every item has been released to the default branch.

## Unreleased

### Added
- Guided notebooks for matrix arithmetic, numerical diagnostics, transformations, and benchmarking.
- Additional tests for matrix validation, numerical methods, and the optional FastAPI API.
- Continuous integration for supported Python versions.
- More complete API, user, architecture, and mathematical documentation.

### Changed
- Expanded the Matrix wrapper with pseudoinverse, condition number, eigenvectors, symmetry checks, copy/export helpers, and more explicit validation.
- Expanded the HTTP API operation set and made complex eigenvalues and infinite condition numbers JSON-safe.
- Rewrote the README and notebook guide.

### Notes
- The implementation continues to delegate numerical algorithms to NumPy.
- Performance results must be measured and reported with reproducible environment details.
