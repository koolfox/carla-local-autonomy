# Evidence Index and Sealed Registry

Status: implemented thesis-development infrastructure  
Mutation policy: read source objects; write one new registry object  
Simulator policy: no CARLA connection or simulator mutation

## Purpose and claim boundary

The evidence registry creates a reproducible, point-in-time inventory of the
research objects in one workspace. It records object provenance, declared
artifacts, verification outcomes, canonical CSV tables, derived plots, and a
human-readable report under one new tracked run.

The registry is not a database, a copy of every payload, or evidence that a
scientific claim is valid. A listed object may be incomplete, failed, or
invalid. The object table keeps its manifest-declared status separate from its
independently computed `verification_ok` result.

The operator UI's Evidence Explorer is a related but different facility. It is
a live, manifest-declared catalogue for browsing registered artifacts; it does
not run hashes or semantic verification. See
[`operator_ui.md`](operator_ui.md).

## Point-in-time scope

At build time, the scanner records the UTF-8-sorted immediate child
directories of these canonical roots:

```text
datasets/<dataset-id>/
runs/<run-id>/
models/<model-id>/
reports/<report-id>/
bundles/<bundle-id>/
native_kits/<kit-id>/
operator_sessions/<job-id>/
```

Missing roots, ordinary non-directory entries, and deeper descendants are not
independently discovered. An immediate-child symlink is retained as invalid
evidence without following its target. An immediate child with a missing,
malformed, non-finite, or symlinked manifest is likewise retained as a
verification failure rather than silently omitted.

The resolved source paths and excluded paths are frozen in
`registry_config.json`. Objects created after the build are therefore outside
that registry's scope and do not make its later verification fail. A changed
or removed recorded source does fail verification. Create a new registry ID
to represent a later workspace state; registry outputs are never overwritten.

## Exclusion and cycle behavior

The output being built and all previously recognized evidence-registry objects
are excluded from discovery. Excluded paths are recorded in the configuration,
summary, descriptor, and report. This prevents registry-of-registry recursion
and keeps repeated snapshots comparable.

This targeted exclusion is not a general proof that every arbitrary provenance
graph is acyclic. Source-specific deep verifiers and fingerprinted-reference
checks still define the integrity boundary for each discovered object.

## Verification behavior

The builder attempts verification for every discovered source with:

```text
verify_references = true
deep = true
reject_unregistered = true
allow_non_success = true
```

Consequently:

- registered file sizes and SHA-256 values, checksum indexes, external
  references, safe paths, and supported object-specific semantics are checked;
- unexpected files and symlinks fail strict verification;
- failed or stopped runs can pass integrity verification when their retained
  evidence is internally consistent;
- verification failures remain in the registry with a bounded error type and
  message; they do not disappear from the inventory;
- regular source manifests are read without following symlinks and
  fingerprinted again after inspection, so a change during the scan is
  reported as a failure;
- each invalid source receives a deterministic non-following tree inventory:
  regular files are hashed, while symlinks record only their path, type, and
  link text. Byte or path drift inside already-invalid evidence therefore
  fails later registry verification.

The registry verifier first checks the registry object's own tracker manifest,
registered payloads, roles, checksums, and unregistered-file boundary. It then
reconstructs the recorded snapshot from the current workspace and requires the
configuration, summary, tables, report, plots, and descriptor to reproduce
byte-for-byte. This detects both registry tampering and drift in recorded
sources. Later, unrecorded workspace additions are intentionally ignored.

## Output tree

```text
runs/<registry-id>/
├── manifest.json
├── registry_config.json
├── registry.json
├── summary.json
├── report.md
├── checksums.sha256
├── tables/
│   ├── objects.csv
│   ├── artifacts.csv
│   └── failed_source_tree.csv
└── plots/
    ├── root_inventory.png
    ├── root_inventory.svg
    ├── verification_status.png
    └── verification_status.svg
```

`objects.csv` is the canonical object-level inventory. `artifacts.csv`
preserves each manifest artifact declaration, including its source, role,
digest, size, media type, and declaration-validity flag. `registry.json`
binds the sources and output fingerprints; `checksums.sha256` covers the
retained registry payloads. `failed_source_tree.csv` freezes the non-following
contents of sources that could not pass verification. The tracker
`manifest.json` registers every generated payload and fingerprints safe,
available source manifests as inputs.

## CLI

From the workspace root, build a new snapshot with a unique lowercase ID:

```bash
uv run carla-build-evidence-index \
  --workspace . \
  --registry-id evidence-registry-20260727-v001
```

Verify its own integrity, its recorded sources, and its reproducibility:

```bash
uv run carla-verify \
  runs/evidence-registry-20260727-v001 \
  --reject-unregistered
```

To capture a later state, use a new ID rather than replacing the first object:

```bash
uv run carla-build-evidence-index \
  --workspace . \
  --registry-id evidence-registry-20260728-v001
```

## Portability limitation

This registry is workspace-bound. Its configuration records the canonical
absolute workspace path, and verification requires the registry at
`<workspace>/runs/<registry-id>` with the recorded source objects still
available there. The registry inventories and fingerprints source manifests;
it does not embed all source payload bytes. Moving only the registry directory
is therefore not a valid portable reproduction package. Use a standalone
reproduction bundle when transport or offline archival is required.

## Safety boundary

Building or verifying a registry does not contact CARLA, reload a map, spawn
or destroy actors, move a vehicle, train a model, or modify any source research
object. The builder writes only a new `runs/<registry-id>/` object and refuses
an existing output path.

“Sealed” means that exact retained bytes and source state are integrity-checked;
it does not mean the local filesystem is physically immutable. Deliberate or
accidental changes are detected by verification rather than prevented by
filesystem permissions. The broader retention and publication requirements
remain defined in [`artifact_policy.md`](artifact_policy.md).
