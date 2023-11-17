# CHANGELOG

All notable changes to `radifox` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
 - Added CHANGELOG.md and filled in previous changes.

## [1.0.2] - 2023-11-15
This version switches the sigpy dependency to a lighter fork (sigpy-lite) that does not automatically depend on numba and pywavelets.
This greatly reduces the size of the dependency packages (and is not needed for this use).
The Dockerfile is also updated to remove unnecessary caches.

## [1.0.1] - 2023-11-13
Inital combined release of `degrade` and `resize` as `radifox-utils`.