---
name: CARLA Garage
description: A design-tool workspace for CARLA research sessions, with the live simulator camera as the artboard.
colors:
  canvas: "#ebebe9"
  panel: "#ffffff"
  panel-2: "#f4f4f2"
  panel-3: "#e9e9e6"
  line: "#e2e2df"
  line-strong: "#cfcfcb"
  text: "#1b1b1a"
  text-2: "#5f5f5b"
  text-3: "#767672"
  accent: "#2f5bea"
  accent-hover: "#2149d4"
  accent-ink: "#1f45c9"
  accent-soft: "#e8edfd"
  accent-ring: "rgba(47, 91, 234, 0.28)"
  ok: "#1d7a4c"
  ok-soft: "#e4f3ea"
  warn: "#95580c"
  warn-soft: "#fbf0dc"
  bad: "#c0362e"
  bad-ink: "#a62c25"
  bad-soft: "#fbe8e6"
  stage: "#151517"
  on-stage: "#f3f3f4"
  on-stage-2: "#b4b4bb"
  scrim: "rgba(20, 20, 23, 0.78)"
  scrim-line: "rgba(255, 255, 255, 0.12)"
typography:
  headline:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: "-0.01em"
  title:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "13.5px"
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: "-0.005em"
  body:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.45
  body-sm:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "11.5px"
    fontWeight: 500
    lineHeight: 1.45
  rail-label:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "10.5px"
    fontWeight: 500
    lineHeight: 1.1
  readout:
    fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  data:
    fontFamily: "ui-monospace, SF Mono, SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace"
    fontSize: "0.92em"
    fontWeight: 400
rounded:
  artboard: "3px"
  segment: "5px"
  sm: "6px"
  group: "7px"
  hud: "8px"
  overlay: "9px"
  lg: "10px"
  sheet: "14px"
  pill: "999px"
spacing:
  hair: "2px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  field: "10px"
  lg: "12px"
  section: "14px"
  xl: "16px"
  control-h: "30px"
  control-h-touch: "40px"
  topbar-h: "46px"
  rail-w: "72px"
  inspector-w: "368px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "#ffffff"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "{spacing.control-h}"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-secondary:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "{spacing.control-h}"
  button-secondary-hover:
    backgroundColor: "{colors.panel-2}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.text-2}"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "{spacing.control-h}"
  button-danger:
    backgroundColor: "{colors.bad}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
    padding: "0 12px"
    height: "{spacing.control-h}"
  button-danger-hover:
    backgroundColor: "{colors.bad-ink}"
  button-disabled:
    backgroundColor: "{colors.panel-3}"
    textColor: "{colors.text-3}"
  field:
    backgroundColor: "{colors.panel-2}"
    textColor: "{colors.text}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: "0 9px"
    height: "{spacing.control-h}"
  field-focus:
    backgroundColor: "{colors.panel}"
  rail-item:
    backgroundColor: "transparent"
    textColor: "{colors.text-2}"
    typography: "{typography.rail-label}"
    rounded: "{rounded.sm}"
    padding: "8px 2px 7px"
  rail-item-active:
    backgroundColor: "{colors.accent-soft}"
    textColor: "{colors.accent-ink}"
  segmented-control:
    backgroundColor: "{colors.panel-2}"
    rounded: "{rounded.group}"
    padding: "2px"
  segmented-item-active:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.segment}"
  option-tile:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "9px 10px"
  option-tile-active:
    backgroundColor: "{colors.accent-soft}"
  floating-toolbar:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.lg}"
    padding: "6px"
  stage-overlay:
    backgroundColor: "{colors.scrim}"
    textColor: "{colors.on-stage}"
    rounded: "{rounded.overlay}"
  notice:
    backgroundColor: "{colors.panel-2}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: "10px 12px"
---

# Design System: CARLA Garage

## Overview

**Creative North Star: "The Artboard Is the Camera"**

The console borrows the grammar of a design tool (Paper): a soft neutral canvas, white tool panels hung on 1px hairlines, a slim icon rail on the left, an inspector on the right, and one floating toolbar under the work. The work itself is the live CARLA camera, framed and labelled like an artboard with its real applied resolution and frame rate. Everything else is tooling around that frame; the camera never gets covered by a modal menu.

Density is high and quiet. UI type sits at 11.5 to 15px in a single system grotesque, every number is tabular, and almost nothing is boxed: sections are separated by hairlines, not cards. Colour is rationed. One ink blue means "selected, focused, or the primary action"; green, amber and red mean state, and state is always carried by the shape of an indicator as well as its hue. A dark theme is derived automatically from the same token names via `prefers-color-scheme`; the camera stage is dark in both themes because it is imagery, not chrome.

**Key Characteristics:**
- Neutral canvas, white panels, 1px hairlines; separation by line, not by card.
- One accent (ink blue) for selection, focus, primary action and the running state.
- State dots whose shape encodes the state: solid, ring, half-filled, struck, dashed, pulsing.
- The camera stage stays dark in light and dark themes; overlays on it share one scrim.
- Shadows only on things that float above the canvas: the artboard, the floating toolbar, primary buttons, the active segment.
- One drawn icon set on a 24-unit grid with a 1.6 stroke.

## Colors

A warm-grey paper workspace with a single ink-blue selection colour and three state colours that never appear without a matching shape.

### Primary
- **Selection Ink** (`accent`): the only interactive accent. Primary buttons, the active rail item and option tile (on `accent-soft`), focus outlines and focus rings (`accent-ring`), checked switches, range thumbs, caret and the pulsing "running" dot. `accent-hover` darkens the primary button on hover; `accent-ink` is the readable text form used on `accent-soft` and for links.

### Tertiary (state)
- **Good Green** (`ok`, `ok-soft`): online, success, ready. Always paired with a solid dot.
- **Pending Amber** (`warn`, `warn-soft`): pending sync, retrying, an applied value that differs from the requested one, autonomy acknowledgement. Paired with a half-filled dot or a dashed underline.
- **Fault Red** (`bad`, `bad-ink`, `bad-soft`): offline, failed, errors, Emergency Brake. Paired with a hollow ring; `bad-ink` is the readable text form.

### Neutral
- **Paper Canvas** (`canvas`): the workspace behind the artboard and the page ground.
- **Panel White** (`panel`): top bar, rail, inspector, floating toolbar, option tiles, secondary buttons.
- **Panel Fill** (`panel-2`): resting fill of inputs and selects, segmented-control track, notices, `pre` blocks, hover fill for quiet controls.
- **Panel Deep** (`panel-3`): disabled buttons and the mode-icon well.
- **Hairline** (`line`) and **Hairline Strong** (`line-strong`): every divider; `line-strong` for secondary-button borders, input hover borders, scrollbar thumbs and the sheet grabber.
- **Ink** (`text`), **Ink Secondary** (`text-2`), **Ink Tertiary** (`text-3`): primary copy, labels and notes, placeholders and disabled text.
- **Stage** (`stage`, `on-stage`, `on-stage-2`): the camera surface and its text. **Scrim** (`scrim`, `scrim-line`): the single translucent material for anything laid over imagery.

### Named Rules
**The One Ink Rule.** Blue means selected, focused, primary or running. It is never decoration, never a heading colour, never a second brand tint.

**The Shape Carries State Rule.** Every state colour ships with a shape: solid dot = good or idle, hollow ring = offline or failed, half-filled ring = pending, struck ring = not configured, dashed ring = waiting or unknown, pulsing solid = running. A screen read in greyscale must still tell the states apart.

**The Stage Stays Dark Rule.** The camera surface uses `stage` in both themes; light-theme chrome never bleeds onto imagery.

## Typography

**Body Font:** system UI grotesque (`ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, …`)
**Label/Mono Font:** system monospace (`ui-monospace, SF Mono, Menlo, Consolas, …`) for run IDs, saved paths and code

**Character:** One workhorse sans at tool scale; hierarchy comes from weight (400/500/600) and ink level, not size jumps. There is no display tier: the largest text is the 19px placeholder headline on an empty stage.

### Hierarchy
- **Headline** (600, 15px, -0.01em): inspector section title, brand-adjacent titles, HUD readout values, loading and empty-stage headlines.
- **Title** (600, 13.5px, -0.005em): section headings inside the inspector, brand name.
- **Body** (400, 13.5px, 1.45): base copy.
- **Body small** (400–600, 12.5px): control text (inputs, buttons, tiles, status pills, breadcrumb, notices).
- **Label** (500, 11.5px): field labels, notes, key-value keys, artboard label, HUD captions. Sentence case.
- **Rail label** (500, 10.5px, 1.1): text under rail icons and HUD captions only.
- **Data** (mono, 0.92em): run IDs, file paths, `code` and `pre`.

### Named Rules
**The Tabular Rule.** Every number that can change (resolution, FPS, speed, RPM, counts, field values) uses `font-variant-numeric: tabular-nums`.

**The Sentence Case Rule.** Labels, headings and buttons are sentence case at their natural tracking. Capitalisation transforms appear only to normalise data values (preset names, map names), never as a label style.

## Layout

The shell is a fixed 100dvh grid: a 46px top bar over a three-column workspace of rail (72px), canvas (fluid) and inspector (368px; 320px at 1366px and below; wide sections such as Vision and Recordings take `min(640px, 48vw)`). The top bar holds brand lockup, a map / run-ID breadcrumb, the clickable status cluster, and the panel toggle at its right edge.

The canvas centres the artboard: a 16:9 stage sized by container queries to the larger of the available width or height, with a label row above (vehicle name left, applied resolution and FPS right) and a HUD row below (map, weather, population, separated by 1px vertical ticks). The floating toolbar sits under the artboard, centred. While driving, the stage fills the canvas edge to edge inside 12px padding and the toolbar floats over it, 24px from the bottom.

Spacing is a small-step rhythm: 2, 4, 6, 8, 10, 12, 14, 16px. Inspector sections pad 16px with 14px internal gaps and a hairline below; fields stack at 12px gaps with 5px between label and control. Field grids collapse 3 → 2 → 1 columns by inspector container width (440px, 300px).

At 860px and below, the rail and inspector merge into one bottom sheet (14px top corners, a 36×4px grabber, rail as a horizontal tab strip, inspector at `56dvh - 66px`) over a 40% black backdrop; the canvas goes full width and the toolbar stretches. On touch devices, controls grow to 40px, thumb keys take the bottom corners, and the drive toolbar moves above them.

## Elevation & Depth

Flat by default, with depth reserved for what floats above the canvas. Panels sit flush against the canvas and are separated only by hairlines; inspector sections are not cards. Overlays on imagery use the translucent scrim with a 10px backdrop blur for legibility over video, never a shadow. The dark theme replaces the artboard's drop with a 1px light ring so the frame still reads against a dark canvas.

### Shadow Vocabulary
- **Lift** (`--shadow-sm`): primary and danger buttons, the active segment of a segmented control.
- **Float** (`--shadow-float`): the floating toolbar (Garage and drive).
- **Artboard** (`--shadow-artboard`): the camera stage, drive viewport and recording player.

### Named Rules
**The Float Only What Floats Rule.** A shadow means "this sits above the canvas." Panels, sections, tiles and notices never take one.

**The One Scrim Rule.** Everything laid over the camera (view switcher, badges, HUD readouts, telemetry, manual-control prompt, cockpit errors) uses `scrim` fill, `scrim-line` border and a 10px blur.

## Shapes

Small, consistent corners. Controls, tiles, notices and quiet buttons use 6px; containers that float use 10px; the artboard uses a near-square 3px so it reads as a frame, not a card. Nested pieces step down one notch (segmented track 7px around 5px segments; mode-icon wells 5px). Overlays on the stage use 9px (8px for HUD readouts). Dots, switches and scrollbar thumbs are fully round; the mobile sheet is the only element with 14px corners.

When the car is grabbed or the stage is keyboard-focused, the artboard is "selected" like a frame on a design canvas: a 1px accent outline and four 7px square handles appear at its corners.

## Components

### Buttons
Quiet and compact; colour is reserved for the one action that matters.
- **Shape:** gently squared (6px), 30px tall (34px in the drive toolbar, 26px compact, 40px on touch), 12px side padding, 500 weight at 12.5px.
- **Primary:** accent fill, white text, Lift shadow; hover to `accent-hover`.
- **Secondary:** white panel with a `line-strong` border; hover fills `panel-2`.
- **Ghost:** transparent, `text-2`; hover fills `panel-2` and inks to `text`.
- **Danger:** `bad` fill, white 600 text, Lift shadow; hover to `bad-ink`. Used for Emergency Brake.
- **Disabled:** every variant collapses to `panel-3` fill with `text-3`, no border, no shadow.
- **Icon button:** 28px square, transparent, `text-2`, `panel-2` hover.

### Segmented Control
A 2px-padded `panel-2` track (7px) of transparent 5px segments in 11.5px label type; the active segment lifts to white with Lift. On the stage the track becomes 6% white and the active segment 16% white with no shadow.

### Option Tiles
Two-column grid of white tiles (6px, 1px `line` border, 9×10px padding) with a 600 title and a `text-2` caption, optionally led by a 26px mode-icon well. Hover strengthens the border; active takes an accent border, an inset 1px accent ring and `accent-soft` fill, and the mode icon inverts to accent.

### Inputs / Fields
- **Style:** borderless fill: `panel-2`, 1px transparent border, 6px corners, 30px tall, 9px side padding, 12.5px tabular text. Selects carry a drawn 12px chevron.
- **Hover:** border to `line-strong`.
- **Focus:** border to `accent`, a 3px `accent-ring` halo, and fill lifts to `panel`.
- **Disabled:** `text-3` at 70% opacity.
- **Switches:** 28×16px track (`line-strong`, accent when on) with a 12px white knob; one row per option, hairlines between rows.

### Status Indicators
8px dots inside 26px pills of 12.5px `text-2` text, shaped per The Shape Carries State Rule. Offline/failed pills ink their text `bad-ink`. Three pills (CARLA, bridge, session) sit together in the top bar as one button that opens System.

### Key–Value Facts
One table grammar everywhere: two columns (`minmax(96px, .8fr)` / `1.4fr`), 7px rows divided by hairlines, label-size `text-2` keys, 500-weight 12.5px tabular values that truncate.

### Notices
6px boxes on `panel-2` with a hairline border, 10×12px padding, a 600 lead line and `text-2` detail. Warning, error and success variants swap to the `-soft` fill with a 45% state-colour border.

### Navigation
The section rail: 72px column of icon-over-label items (18px icon, 10.5px label), transparent at rest, `panel-2` on hover, `accent-soft` with `accent-ink` when active, 38% opacity when disabled. On mobile it becomes the horizontal tab strip at the top of the bottom sheet.

### Floating Toolbar (signature)
The design tool's bottom bar: a white 10px container with a hairline border, Float shadow and 6px padding, holding vehicle carousel, paint picker, camera-preset segmented control, preview state and the session action. While driving the same toolbar holds Emergency Brake and Stop & Save, and on touch it parks above the thumb keys.

### Artboard (signature)
The camera stage: 16:9, 3px corners, `stage` fill, Artboard shadow, labelled above and annotated below. When an applied value differs from the request (resolution, population), that value turns `warn` with a dashed underline until it matches.

### Icons
One drawn set: 24-unit viewBox, 1.6 stroke, round caps and joins, `currentColor`, 16px default (18px in the rail, 20–22px on thumb keys).

## Do's and Don'ts

### Do:
- **Do** separate inspector content with 1px `line` hairlines and 16px section padding.
- **Do** pair every state colour with its dot shape (solid, ring, half, struck, dashed, pulse).
- **Do** keep the camera stage on `stage` in both themes and put every overlay on the shared scrim with a 10px blur.
- **Do** use tabular numerals for every changing number.
- **Do** mark a held, not-yet-applied or adjusted value in `warn` (half-filled dot or dashed underline) until it is applied.
- **Do** draw new icons in the shared set: 24-unit grid, 1.6 stroke, round joins.
- **Do** derive dark-theme values by remapping the same token names, never by new component styles.

### Don't:
- **Don't** use the accent blue for anything other than selection, focus, primary action or the running state.
- **Don't** signal state by colour alone.
- **Don't** wrap inspector sections in shadowed or bordered cards; only floating things take a shadow.
- **Don't** cover the camera with a modal menu; settings live in the inspector or the bottom sheet.
- **Don't** use uppercase tracked labels or eyebrow text above headings.
- **Don't** hard-code colour literals in components; extend the token set in `app.css` instead.
