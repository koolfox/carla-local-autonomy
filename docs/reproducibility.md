# Reproducibility and Provenance

This document defines the implemented provenance boundary for CARLA Vision
framework version 0.4. It complements the normative experiment and artifact
policies; it does not turn a development run into confirmatory evidence.

## 1. Canonical configuration identity

`carla_vision.reproducibility` implements a frozen
`json-sort-keys-utf8-v1` serialization:

- mappings are serialized with lexicographically sorted string keys;
- arrays retain order;
- UTF-8 is retained without ASCII escaping;
- insignificant JSON whitespace is removed;
- non-finite floats and unsupported Python values are rejected;
- negative floating-point zero is normalized to zero;
- aware datetimes are normalized to UTC with a trailing `Z`.

Every new tracker manifest records two SHA-256 identities:

- `resolved_sha256` covers the complete redacted resolved configuration;
- `identity_sha256` excludes only the frozen top-level allocation fields
  `created_at`, `started_at`, `run_id`, `runs_root`, `output_dir`, and
  `output_path`.

The resolved digest detects every retained configuration change. The identity
digest allows an experiment to move between output directories without
changing its scientific identity. Sensitive configuration values are redacted
before either digest is computed. Credentials must never be scientific
parameters; two configurations that differ only by a secret therefore share
the same manifest identity.

The verifier recomputes both digests from `invocation.config`. Editing the
configuration or either digest invalidates the object even if an attacker
rewrites `manifest.json`.

## 2. Canonical experiment IDs

The shared helper produces:

```text
exp-<UTC>-<stage>-<model>-<cfg8>-s<seed>
```

`UTC` is the allocation/start timestamp, `cfg8` is the first eight characters
of `identity_sha256`, and the seed must be in `[0, 2^31)`. Stage names are
restricted to a versioned vocabulary. Model and stage labels are normalized to
lowercase hyphenated slugs.

`RunArtifactTracker` can allocate this ID directly when
`experiment_stage`, `experiment_model`, and `master_seed` are supplied
together. If an explicit `run_id` is also supplied, it must equal the
recomputed canonical ID. The verifier reconstructs the ID from the manifest
timestamp, configuration, stage, model, and seed.

Legacy schema-1.0 artifacts created before this extension remain verifiable.
Their absent `reproducibility` field is treated as disclosed missing
provenance, never fabricated after the fact.

## 3. Dependency and hardware envelope

New manifests fingerprint these repository-relative dependency locks when
present:

- `uv.lock`;
- `poetry.lock`;
- `Pipfile.lock`;
- `requirements.txt`.

Only the relative name, SHA-256, and byte size enter the run manifest. A
regular run does not depend on the mutable original lockfile remaining at the
same path. Reproduction bundles archive the complete lockfile bytes.

The privacy-safe hardware allow-list records:

- CPU architecture, model, and logical core count;
- total physical memory;
- PyTorch version and accelerator availability;
- CUDA runtime, enumerated CUDA device names, ROCm, cuDNN, and MPS state;
- deterministic-algorithm, cuDNN benchmark, and cuDNN deterministic flags;
- Python compiler, Git version, and uv version;
- container detection and an image digest when a higher-level runtime supplies
  one.

It never reads or serializes environment variables, hostname, username,
network interfaces, repository remotes, or arbitrary custom-probe fields.
The verifier enforces the complete allow-listed schema. Device and tool values
describe the client that created the artifact; verification does not require
the consumer to have identical hardware.

## 4. Standalone reproduction bundle

A strict JSON preregistration declares:

- immutable bundle ID, title, authors, and development/confirmatory purpose;
- repository root;
- verified research objects and their semantic roles;
- exact source-tree paths to archive;
- tokenized reproduction commands;
- explicit limitations.

Build and verify a bundle with:

```bash
uv run carla-build-reproduction \
  --config configs/reproduction/framework_development_bundle_v1.json \
  --bundles-root bundles

uv run carla-verify-reproduction \
  bundles/bundle-framework-development-20260726-v001

uv run carla-verify \
  bundles/bundle-framework-development-20260726-v001 \
  --reject-unregistered
```

The builder first recursively verifies every source object, including its
external fingerprints and semantic contract. It then writes:

```text
manifest.json
bundle.json
bundle_config.json
checksums.sha256
commands.json
reproduce.sh
README.md
environment/environment.json
environment/lockfiles/*
source/source.zip
source/inventory.json
source/inventory.csv
sources/source_graph.json
sources/<source-id>/manifest.json
```

The source archive uses uncompressed ZIP entries with a fixed 1980 timestamp,
sorted UTF-8 paths, normalized `0644`/`0755` modes, and no symlinks. The
inventory retains SHA-256, byte size, and archive mode for every file. Files
with credential/key suffixes, environment-secret names, cache directories, or
unsafe paths cannot enter the archive. Per-file and total byte limits prevent
accidental packaging of model weights or datasets as source code.

Commands are arrays of tokens, not free-form shell snippets. The generated
script shell-quotes every token, refuses to overwrite an existing workspace,
extracts the source snapshot, and executes the frozen sequence.

## 5. Semantic verification

The dedicated verifier checks more than outer hashes:

- descriptor references point to the artifact registered for the expected
  role;
- JSON and CSV source inventories are identical;
- every ZIP member reproduces the inventory digest, size, order, timestamp,
  type, and mode;
- every archived source manifest matches its source-graph fingerprint and
  retained verification evidence;
- dependency-lock copies match the tracker provenance;
- environment and hardware snapshots match the tracker envelope;
- tokenized commands match the frozen config;
- `reproduce.sh` is executable and regenerates byte-for-byte from the command
  arrays;
- descriptor counts and limitations reconcile;
- the checksum index covers every payload and there are no unregistered files.

Bundle verification has zero dependency on the original source directories.
The original datasets, weights, videos, or CARLA binaries are not silently
embedded. If they are needed for an empirical rerun, their source manifests
provide the required digest and retrieval identity.

## 6. Confirmatory gate

A confirmatory bundle additionally requires:

- an available Git commit and `dirty: false` for the bundle source;
- clean-Git evidence for every archived research object;
- at least one dependency lock;
- an explicit `uv sync --frozen` reproduction command;
- all ordinary object, source-reference, semantic, checksum, and unregistered
  file checks.

The current repository has no commit and remains dirty. Therefore the retained
bundle is deliberately labeled `development`; it must not be cited as a
confirmatory thesis result.

## 7. Retained development evidence

`bundle-framework-development-20260726-v001` contains:

- 10 recursively verified source research objects;
- 143 source files and their canonical inventory;
- one archived `uv.lock`;
- four tokenized reproduction commands;
- 22 registered artifacts and a 21-entry checksum index;
- zero fingerprinted external dependencies for bundle verification;
- zero unregistered files.

Its source archive predates later v0.4 documentation edits. A newer bundle ID
must be created for the final clean baseline; the existing ID is never
overwritten or retroactively relabeled.
