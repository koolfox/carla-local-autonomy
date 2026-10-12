# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

The established product is a browser console. Its requested launch surface is a
full-screen terminal application for Windows first and Linux alongside it.

## Users

End users who have CARLA on the same computer and want to install, start and use
the console without manually coordinating Python environments and processes.

## Product Purpose

Run the CARLA server, authenticated World Worker and existing browser console on
one computer, with a self-installing launcher, model assets and clear readiness.

## Operating Context

CARLA 0.9.16, Windows x86-64 first and Linux x86-64 support. The browser UI stays
the primary interaction surface for Garage, driving, perception and recordings.
The terminal manages installation, local service lifecycle and diagnostics.

## Capabilities and Constraints

- Full-screen TUI requested by the maintainer on 2026-10-12.
- Native same-host setup targets Windows and Linux; existing remote topology remains supported.
- Retain the existing web UI, Worker API, actor ownership and model contracts.
- Prefer installed CARLA; support deliberate official-package download when missing.
- Loopback connections and generated local credentials for single-host operation.
- Runtime and live-CARLA validation are deferred; no new validation claim is permitted.
- New work uses a feature branch and remains subject to maintainer-owned PR and merge.

## Evidence on Hand

Existing source, compiled Svelte console, pinned dependency lock, ten model
weights in Git LFS, class tables and historical sample material. No live
Windows/Linux launcher acceptance is available at implementation time.

## Product Principles

- Install once; start the local stack from one surface.
- Readiness comes from actual local APIs, not a listening port alone.
- Stop only services started by the launcher; preserve external simulator processes.
- Keep simulator and perception execution behind the existing product boundaries.
