# CARLA Vision Framework Validation Report v0.9

Date: 2026-07-27  
Status: evidence-index and operator-explorer milestone complete  
Primary sensor boundary: one forward-facing monocular RGB camera  
CARLA endpoint: `172.20.10.7:2000`

## Outcome

The thesis MVP now has a repository-wide evidence layer without introducing a
database or a second orchestration backend. A sealed point-in-time registry
inventories all canonical research roots, runs strict verification on every
discovered object, retains failures rather than hiding them, and produces
machine-readable tables, plots, checksums, and a human-readable report.

The local operator panel now has a fifth surface, **Evidence**, for searching
and inspecting manifest-declared artifacts. It does not claim that catalogue
entries are verified. Operators explicitly hand a selected object to the
existing read-only verifier.

No CARLA connection, simulator mutation, model training, or source-object
modification was performed by this milestone.

## Sealed workspace index

```text
runs/evidence-index-workspace-20260727-v001
```

| Measure | Value |
|---|---:|
| Canonical roots | 7 |
| Discovered source objects | 47 |
| Source manifests fingerprinted | 47 |
| Strictly verified objects | 46 |
| Verification failures retained | 1 |
| Registered artifacts inventoried | 437 |
| Declared artifact bytes | 44,710,909 |
| Invalid-source tree entries | 10 |
| Invalid-source regular files | 8 |
| Invalid-source regular-file bytes | 1,388,800 |
| Clean-Git objects | 0 |
| Dirty-Git objects | 47 |
| Duplicate object-ID groups | 0 |
| Simulator contacted | false |
| Simulator mutated | false |
| Source objects modified | false |

Root counts at snapshot time were:

```text
datasets 1
runs 24
models 0
reports 7
bundles 7
native_kits 3
operator_sessions 5
```

The one retained failure is the historical
`native-host-kit-pilot-20260727-v001`. Its current semantic verifier reports
that the payload plan does not reproduce from its configuration and scenario
plan. The object was not changed or suppressed. Its non-following tree
snapshot preserves ten entries so further byte or path drift is detectable.
The corrected v002 and final v003 kits both verify.

Registry manifest SHA-256:

```text
8a8426b70e182651c81d5959465d5ffc76b906c2dfaae3a5216b456f8f7b0738
```

Registry descriptor SHA-256:

```text
d2dff55dc3671717c1449f04dbe653ce61924dbbeb6b0705cd8d8f7dd9da51cd
```

Generic deep verification reports 12 registered registry artifacts, 11
checksum entries, 47 verified external manifest references, zero unregistered
files, and `kind: evidence_registry`.

## Failure-preserving and path-safe behavior

The adversarial regression suite proves that:

- a symlinked canonical root is rejected before output creation;
- immediate-child, dangling-child, and manifest symlinks are retained as
  failed evidence without dereferencing their targets;
- external symlink targets never enter registry input references;
- malformed JSON and non-finite `NaN`, `Infinity`, and `-Infinity` manifests
  are retained with hashes of their raw bytes;
- an unregistered file or missing-manifest source receives a deterministic
  non-following tree inventory;
- later byte changes inside already-invalid evidence fail registry
  verification;
- later unrelated objects do not invalidate an older point-in-time snapshot;
- registry payload tampering and recorded valid-source drift are rejected.

This is deliberately narrower than a graph database: registries are excluded
from registry discovery to avoid recursive snapshots, but no global
cross-object cycle or supersession service is claimed.

## Evidence Explorer

The loopback UI at `http://127.0.0.1:8765/` now:

- lists manifest-backed objects across all seven roots;
- filters by free text, root, and manifest-declared status;
- shows declared role, path, size, SHA-256, metadata, and current availability;
- previews registered PNG/JPEG/SVG and MP4 artifacts;
- safely downloads registered ZIP, model, YAML, and checksum artifacts;
- rejects traversal, drive paths, symlinked objects/files, missing files, and
  unregistered paths;
- applies a script-disabled browser policy to research artifact responses;
- selects the exact object in the existing verification workflow;
- derives Session links and previews from manifests rather than guessed
  overlay/video names.

The UI consistently labels this live catalogue `not_checked`. File
availability is not hash or semantic verification.

## Automated evidence

```text
181 passed, 50 subtests passed
Ruff formatting passed
Ruff lint passed
Python byte-code compilation passed
JavaScript syntax passed
uv lock consistency passed
focused HTTP security tests passed
in-app browser interaction and console checks passed
```

## Manifest-generated report

The four-source report is:

```text
reports/rpt-framework-validation-20260727-v008
```

It contains 33 canonical metric rows, 45 source-artifact rows, nine registered
artifacts, four verified external source references, and no unregistered
files. Evidence-index metrics include all object, failure, artifact,
failed-tree, exclusion, and no-mutation values without manual transcription.

Report manifest SHA-256:

```text
757de2b1871ed6cc4e505f75d1877ff5d93123572619a8fe30274f3e24971ee0
```

## Standalone reproduction

The final corrected v0.9 source snapshot is configured as:

```text
bundles/bundle-framework-development-20260727-v009
```

Its tokenized commands install the frozen environment, run the complete test,
format, lint, and compile gates, verify the native scenario plan, build and
semantically verify a fresh evidence registry, and check the operator CLI.
The bundle remains development-only and deliberately omits CARLA binaries,
model weights, datasets, and live videos.

The clean execution of superseded bundle v008 passed all commands but exposed
an identity error in its command list: it rebuilt historical kit ID `v003`
from the newer v0.9 source, producing different bytes under an already-used
immutable ID. No retained kit was overwritten. Bundle v008 is preserved as
development defect evidence. The corrected v009 bundle does not regenerate a
historical kit ID; exact v003 reconstruction remains covered by the original
v0.8 bundle, and any future host kit must receive a new configuration and ID.

## Claim boundary and next gate

This evidence supports that the project can discover, inspect, verify, report,
and reproduce its development artifacts through one consistent manifest
contract. It does not convert dirty-worktree artifacts into confirmatory
evidence and does not show that native CARLA collection, human label QA,
RT-DETR training, locked-test evaluation, or vision-only driving succeeds.

The next high-value milestone is not more framework plumbing. It is the
manual 50-frame native pilot on a compatible CARLA 0.9.16 PythonAPI host,
followed by deterministic human QA and dataset release signoff. Only then
should the project advance to a bounded offline vision-policy benchmark and
real RT-DETR training.
