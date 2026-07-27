# CARLA Vision Framework Validation Report v0.8

Date: 2026-07-27  
Status: implementation handoff; manual CARLA acceptance pending  
Primary sensor boundary: one forward-facing monocular RGB camera  
CARLA endpoint: `172.20.10.7:2000`

## Outcome

The native integration implementation is wrapped for hand testing. A
deterministic portable kit now moves the bounded 50-frame pilot to a
Windows/Linux x86-64 host without requiring the full development workspace.
It includes the verified scenario plan, exact collection source, hashed
CPython 3.12 dependencies, read-only preflight, guarded collection, post-run
QA, and both PowerShell and shell entry points.

No real CARLA mutation or native collection was performed for this milestone.
The existing endpoint remains reachable, while the Apple arm64 environment
still lacks the official PythonAPI required only by native synchronous
collection. Live viewing and detector overlays remain on the independent
MessagePack path.

## Portable artifact

```text
native_kits/native-host-kit-pilot-20260727-v003
```

| Field | Value |
|---|---:|
| Target | CPython 3.12, Windows/Linux x86-64 |
| CARLA | 0.9.16 |
| Episodes | 1 train |
| Planned captures | 50 |
| Payload files | 114 |
| Project source files | 95 |
| Payload bytes | 1,322,086 |
| Dependency packages | 14 |
| CARLA wheel hashes | 6 |
| Simulator contacted while building | false |
| Simulator mutated while building | false |

ZIP SHA-256:

```text
185779504f5d789f6dd83eba76728814797512ba854ba63b2957547d3d7c377b
```

Outer manifest SHA-256:

```text
ec0b81358d93aa2edf71e58c6b272d7f3fcb696121356bb1ba8526c01e939165
```

## Execution gates

The host scripts refuse collection unless:

1. the kit payload passes its internal checksum index;
2. the host is CPython 3.12 on x86-64;
3. every installed requirement matches the frozen hashes;
4. a read-only preflight is semantically verified;
5. that preflight reports overall readiness;
6. Dataset ID, endpoint, plan fingerprint, episode order, and capture count
   exactly match the kit;
7. the operator supplies
   `I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK`;
8. the target Dataset directory does not already exist.

The verifier independently checks archive order, fixed timestamps and
permissions, payload hashes/inventories, scenario-plan recomputation, Python
compilation, requirements pins, generated scripts, and safety disclosures.

## Implementation defect closed

The portable-script test exposed that several modules defined `main()` but
lacked a module entry-point guard. Consequently `python -m ...` could exit
successfully without doing work. Guards were added to native preflight,
collection, host gate, verification, and Dataset QA, and the extracted
payload test now exercises the worker dry-run through the exact module
invocation used by the scripts.

## Automated evidence

```text
167 passed, 44 subtests passed
Ruff lint passed
Python byte-code compilation passed
uv lock consistency passed
specialized native-kit verification passed
generic recursive verification passed
```

The host-kit tests also build the package twice and compare bytes, extract it
away from the source plan, run its standalone checksum verifier, syntax-check
the shell scripts, execute the native worker dry-run, and reject archive
tampering and mismatched/not-ready preflights.

## Manifest-generated report

The five-source report is:

```text
reports/rpt-framework-validation-20260727-v007
```

It contains 43 canonical metric rows, 42 source-artifact rows, nine registered
artifacts, five verified external source references, and no unregistered
files. Its manifest SHA-256 is:

```text
01bef3ae1389f673f432dc55838e453657e31c655b4d8c26690ae6140a6f9c09
```

## Claim boundary

The evidence supports that the software can prepare, bind, inspect, and
tamper-check a bounded native-host collection package without touching CARLA.
It does not show that the destination host can install the locks, that real
actors and sensors spawn correctly, that exact RGB/instance delivery succeeds,
that cleanup/restoration survives a real run, or that an RT-DETR checkpoint
has been trained on the resulting data.

The next action is therefore the manual sequence in
[`native_host_kit.md`](native_host_kit.md), followed by human QA of the 50
captured frames. Scaling, training, evaluation, and any vision-only control
remain gated on that acceptance.
