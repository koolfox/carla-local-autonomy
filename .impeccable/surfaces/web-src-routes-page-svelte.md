---
version: 1
slug: "web-src-routes-page-svelte"
primary_target: "web/src/routes/+page.svelte"
related_targets: ["web/src/lib/components","web/src/lib/styles/app.css"]
---

# Garage console (web/)

Scope: the whole Svelte browser console — Garage preview, inspector sections (Experiment, Scene, Vehicle, Vision, Research, Recordings, System), Drive cockpit. Mode: Operate.

Audience/job: researchers running CARLA sessions on laptop and desktop (long sessions, sometimes a dim room), occasionally phone/tablet as a remote monitor or driving pad. Task: watch the live camera/car, adjust scene, vehicle and vision settings, start/stop sessions, review recordings. The live camera is the most prominent element.

Constraints: retain every behavior, API call, copy fact and test-guarded string; no new runtime dependency; Svelte source only, bundle rebuilt. The incumbent was built quickly in many steps and is visually inconsistent; consistency is a primary goal.

## Direction contract

THESIS: The console is a design tool whose artboard is the live CARLA camera. Refuses the modal-menu-over-fullscreen-video arrangement and the generic dark dashboard.

OWN-WORLD: Paper (paper.design) design-tool grammar. Soft neutral canvas, white panels separated by 1px hairlines, 12–13px UI type in one workhorse grotesque with tabular numerals, 6px corners, a single ink-blue selection colour, state colours green/amber/red that are also carried by shape (solid, ring, struck) not hue alone. Automatic dark theme from the same tokens.

STORY: The visitor sees the car immediately, knows readiness from the top bar, changes settings in the inspector while the camera stays visible, and starts or stops from one floating toolbar (while driving it holds Emergency Brake and Stop & Save; on touch devices it sits above the thumb keys; confirmed by the user after finish review).

FIRST VIEWPORT: 44px top bar (brand, map/run breadcrumb, status). Left rail of sections, centered artboard with frame label and real resolution/FPS tag, right inspector ~360px with the active section, floating bottom toolbar with vehicle, paint, camera presets and the primary action. Mobile: canvas full width, inspector as bottom sheet.

FORM: Pinned by user (Paper design tool), overriding roll position 4; seed key 52c00e68. Raises: state readable without colour (emission-line rail); a change holds its highlight until applied (boarding pass); one strict hairline system (Japanese high-density).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
