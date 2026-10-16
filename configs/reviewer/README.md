# Reviewer evidence templates

These are intentionally incomplete, not sample evidence or pre-approved results.
`null`, an empty record array, or `approved=false` must not be replaced with invented
measurements, reviewers, licenses, or model improvements.

Copy the templates outside the repository, fill them using actual retained source
artifacts, and use `fingerprint_file(path)` for each reference. Paths in references
are paired with SHA-256 and byte size. The executable schemas and worked **software
fixtures** are in `reproduction/reviewer.py` and `tests/test_reviewer_workflows.py`.
The complete protocol for selection is `scenarios/study.py`, not this template.

Use [the reviewer runbook](../../docs/reviewer_evidence.md). Live implementation
status belongs to issues #140 and #142, not these static schemas.
