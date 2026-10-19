---
name: Bug report
about: Report a reproducible build, indexing, search, or storage problem
---

## Problem and expected behavior

Describe what happened and what you expected.
Do not post vulnerabilities here; follow SECURITY.md for private reporting.

## Revision and environment

- SPTAG branch and commit:
- OS / architecture:
- CPU features (AVX2 / AVX512 if relevant):
- Compiler, CMake, Boost, TBB, SWIG versions:
- Index algorithm, vector type/dimension, and distance:
- Storage backend and optional integrations:
- TiKV/PD or other dependency versions, if applicable:

## Reproduction

Provide the exact configure/build/run commands, a minimal configuration and
small shareable input, and all relevant environment variable names/values
(redact credentials, private hostnames, and sensitive data).

## Actual output

Include relevant error logs and exit codes as text. If it is a crash, include
a stack trace and whether it reproduces in a baseline configuration.

## Impact and additional context

State whether this is a regression, affects persisted data, or occurs only
with a particular concurrency level or optional backend.
