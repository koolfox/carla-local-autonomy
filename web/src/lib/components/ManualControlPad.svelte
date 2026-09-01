<script lang="ts">
  import { onMount } from 'svelte';

  import { isDriveRunning, speedMetresPerSecond } from '$lib/domain/runtime';
  import {
    garageRuntime,
    sendManualDriveControl,
    takeManualControl
  } from '$lib/stores/runtime';

  export let armRequest = 0;

  type DriveKey = 'forward' | 'brake' | 'left' | 'right' | 'handBrake' | 'reverseModifier';
  type DriveKeys = Record<DriveKey, boolean>;

  const emptyKeys = (): DriveKeys => ({
    forward: false,
    brake: false,
    left: false,
    right: false,
    handBrake: false,
    reverseModifier: false
  });

  let focused = false;
  let keys = emptyKeys();
  let touchKeys = emptyKeys();
  let touchPointers = new Map<number, { control: DriveKey; element: HTMLElement }>();
  let touchForwardSuppressed = false;
  let sequence = 0;
  let inFlight = false;
  let pending = false;
  let lastSafetyStopAt = 0;
  let handledArmRequest = armRequest;

  $: drive = $garageRuntime.drive;
  $: running = isDriveRunning(drive);
  $: extensionOwnsControl = ['behavior', 'imitation', 'voxel'].includes(
    String(drive.garage_mode ?? 'manual').toLowerCase()
  );
  $: autopilot = drive.control_mode === 'autopilot';
  $: emergencyLatched = drive.control_source === 'emergency_stop';
  $: manualAvailable = running && !extensionOwnsControl && !emergencyLatched;
  $: combined = combineKeys();
  $: command = currentCommand();
  $: if (armRequest !== handledArmRequest) {
    handledArmRequest = armRequest;
    if (manualAvailable) void focusControl();
  }

  function combineKeys(): DriveKeys {
    return {
      forward: keys.forward || touchKeys.forward,
      brake: keys.brake || touchKeys.brake,
      left: keys.left || touchKeys.left,
      right: keys.right || touchKeys.right,
      handBrake: keys.handBrake || touchKeys.handBrake,
      reverseModifier: keys.reverseModifier || touchKeys.reverseModifier
    };
  }

  function currentCommand() {
    if (!running || !focused || emergencyLatched || autopilot || extensionOwnsControl) {
      return { throttle: 0, steer: 0, brake: 1, hand_brake: false, reverse: false };
    }
    const active = combineKeys();
    const reverseRequested = active.reverseModifier && active.forward;
    const speed = Math.abs(speedMetresPerSecond(drive));
    const alreadyInReverse = Number(drive.telemetry?.gear) === -1;
    const reverse = reverseRequested && (alreadyInReverse || speed <= 0.5);
    let throttle = active.forward ? (reverse ? 0.35 : 0.55) : 0;
    let brake = active.brake ? 0.85 : 0;
    let handBrake = active.handBrake;
    let steer = Number(active.right) * 0.7 - Number(active.left) * 0.7;
    if (reverseRequested && !reverse) {
      throttle = 0;
      brake = Math.max(brake, 0.65);
    }
    if (brake > 0.01) throttle = 0;
    if (handBrake) {
      throttle = 0;
      brake = 1;
      steer = 0;
    }
    return { throttle, steer, brake, hand_brake: handBrake, reverse };
  }

  function nextSequence(): number {
    const wallClock = Date.now() * 1000 + Math.floor((performance.now() % 1) * 1000);
    sequence = Math.max(sequence + 1, wallClock);
    return sequence;
  }

  async function sendControl(safety = false, keepalive = false): Promise<void> {
    const sessionId = drive.session_id;
    if (!sessionId || extensionOwnsControl) return;
    if (!safety && (autopilot || !running || !focused)) return;
    if (!safety && inFlight) {
      pending = true;
      return;
    }
    const value = safety
      ? { throttle: 0, steer: 0, brake: 1, hand_brake: false, reverse: false }
      : currentCommand();
    if (!safety) inFlight = true;
    try {
      await sendManualDriveControl(
        { session_id: sessionId, sequence: nextSequence(), ...value },
        keepalive
      );
    } catch {
      // Runtime polling surfaces connection failures; local deadman state remains conservative.
    } finally {
      if (!safety) {
        inFlight = false;
        if (pending) {
          pending = false;
          void sendControl();
        }
      }
    }
  }

  function clearKeys(): void {
    for (const [pointerId, item] of touchPointers) {
      try {
        if (item.element.hasPointerCapture(pointerId)) {
          item.element.releasePointerCapture(pointerId);
        }
      } catch {
        // The browser may already have released capture during teardown.
      }
    }
    keys = emptyKeys();
    touchKeys = emptyKeys();
    touchPointers = new Map();
    touchForwardSuppressed = false;
  }

  function releaseControl(sendBrake = true): void {
    focused = false;
    clearKeys();
    if (sendBrake && running && !extensionOwnsControl) {
      const now = Date.now();
      if (now - lastSafetyStopAt > 80) {
        lastSafetyStopAt = now;
        void sendControl(true, true);
      }
    }
  }

  async function focusControl(): Promise<void> {
    if (!manualAvailable) return;
    if (autopilot && !(await takeManualControl())) return;
    focused = true;
    void sendControl();
  }

  function editableTarget(target: EventTarget | null): boolean {
    return target instanceof Element && Boolean(
      target.closest('input, select, textarea, [contenteditable="true"]')
    );
  }

  function keyName(event: KeyboardEvent): DriveKey | null {
    const key = event.key.toLowerCase();
    if (key === 'w' || key === 'arrowup') return 'forward';
    if (key === 's' || key === 'arrowdown') return 'brake';
    if (key === 'a' || key === 'arrowleft') return 'left';
    if (key === 'd' || key === 'arrowright') return 'right';
    if (event.code === 'Space') return 'handBrake';
    if (key === 'shift') return 'reverseModifier';
    return null;
  }

  function handleKey(event: KeyboardEvent, pressed: boolean): void {
    if (!running) return;
    const name = keyName(event);
    if (!name) return;
    if (extensionOwnsControl) {
      event.preventDefault();
      return;
    }
    if (autopilot) {
      event.preventDefault();
      if (pressed) void focusControl();
      return;
    }
    if (!focused) return;
    event.preventDefault();
    if (name === 'reverseModifier' && keys.forward) keys = { ...keys, forward: false };
    keys = { ...keys, [name]: pressed };
    void sendControl();
  }

  function syncTouchKeys(): void {
    const next = emptyKeys();
    for (const item of touchPointers.values()) next[item.control] = true;
    if (touchForwardSuppressed) next.forward = false;
    touchKeys = next;
  }

  async function pointerDown(event: PointerEvent, control: DriveKey): Promise<void> {
    if (!manualAvailable) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    event.preventDefault();
    if (autopilot) {
      await focusControl();
      return;
    }
    if (!focused) await focusControl();
    const element = event.currentTarget as HTMLElement;
    const forwardAlreadyHeld = [...touchPointers.values()].some((item) => item.control === 'forward');
    if (control === 'reverseModifier' && forwardAlreadyHeld) touchForwardSuppressed = true;
    if (control === 'forward' && !forwardAlreadyHeld) touchForwardSuppressed = false;
    try {
      element.setPointerCapture(event.pointerId);
    } catch {
      return;
    }
    touchPointers.set(event.pointerId, { control, element });
    syncTouchKeys();
    void sendControl();
  }

  function pointerUp(event: PointerEvent): void {
    const released = touchPointers.get(event.pointerId);
    if (!released) return;
    event.preventDefault();
    touchPointers.delete(event.pointerId);
    try {
      if (released.element.hasPointerCapture(event.pointerId)) {
        released.element.releasePointerCapture(event.pointerId);
      }
    } catch {
      // Pointer capture may already have been released.
    }
    const forwardStillHeld = [...touchPointers.values()].some((item) => item.control === 'forward');
    if (released.control === 'reverseModifier' && forwardStillHeld) touchForwardSuppressed = true;
    if (released.control === 'forward' && !forwardStillHeld) touchForwardSuppressed = false;
    syncTouchKeys();
    void sendControl();
  }

  function percent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }

  function preventContextMenu(event: MouseEvent): void {
    event.preventDefault();
  }

  onMount(() => {
    const interval = window.setInterval(() => void sendControl(), 67);
    const release = () => releaseControl();
    const keyDown = (event: KeyboardEvent) => {
      if (!editableTarget(event.target)) handleKey(event, true);
    };
    const keyUp = (event: KeyboardEvent) => handleKey(event, false);
    const pointerDown = (event: PointerEvent) => {
      if (
        focused &&
        event.target instanceof Element &&
        !event.target.closest('.drive-viewport, .manual-control-panel')
      ) {
        releaseControl();
      }
    };
    const visibility = () => {
      if (document.hidden) releaseControl();
    };
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', release);
    window.addEventListener('pagehide', release);
    document.addEventListener('pointerdown', pointerDown, true);
    document.addEventListener('visibilitychange', visibility);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', release);
      window.removeEventListener('pagehide', release);
      document.removeEventListener('pointerdown', pointerDown, true);
      document.removeEventListener('visibilitychange', visibility);
      releaseControl(false);
    };
  });
</script>

<section class="manual-control-panel" class:armed={focused} class:disabled={!manualAvailable}>
  <div class="manual-control-heading">
    <span class="manual-control-state">
      {extensionOwnsControl
        ? 'Policy control'
        : autopilot
          ? 'Traffic Manager'
          : focused
            ? 'Keyboard active'
            : 'Click camera to drive'}
    </span>
    {#if autopilot}
      <button
        type="button"
        class="button compact-button"
        disabled={!manualAvailable || $garageRuntime.action === 'takeover'}
        onclick={focusControl}
      >
        Take control
      </button>
    {/if}
  </div>

  <div class="control-surface" aria-label="CARLA manual driving controls">
    <div class="control-readout" aria-label="manual control output">
      <span>T {percent(command.throttle)}</span>
      <span>S {percent(command.steer)}</span>
      <span>B {percent(command.brake)}</span>
    </div>

    <div class="control-keys" aria-label="touch driving controls">
      <button class="touch-left" class:active={combined.left} aria-label="Steer left" onpointerdown={(event) => pointerDown(event, 'left')} onpointerup={pointerUp} onpointercancel={pointerUp} onlostpointercapture={pointerUp} oncontextmenu={preventContextMenu}><strong>←</strong><small>Left</small></button>
      <button class="touch-right" class:active={combined.right} aria-label="Steer right" onpointerdown={(event) => pointerDown(event, 'right')} onpointerup={pointerUp} onpointercancel={pointerUp} onlostpointercapture={pointerUp} oncontextmenu={preventContextMenu}><strong>→</strong><small>Right</small></button>
      <button class="touch-brake" class:active={combined.brake} aria-label="Brake" onpointerdown={(event) => pointerDown(event, 'brake')} onpointerup={pointerUp} onpointercancel={pointerUp} onlostpointercapture={pointerUp} oncontextmenu={preventContextMenu}><strong>■</strong><small>Brake</small></button>
      <button class="touch-throttle" class:active={combined.forward} aria-label="Throttle" onpointerdown={(event) => pointerDown(event, 'forward')} onpointerup={pointerUp} onpointercancel={pointerUp} onlostpointercapture={pointerUp} oncontextmenu={preventContextMenu}><strong>▲</strong><small>Go</small></button>
      <button class="touch-reverse" class:active={combined.reverseModifier} aria-label="Hold for reverse" onpointerdown={(event) => pointerDown(event, 'reverseModifier')} onpointerup={pointerUp} onpointercancel={pointerUp} onlostpointercapture={pointerUp} oncontextmenu={preventContextMenu}><strong>R</strong><small>Reverse</small></button>
      <button class="touch-handbrake" class:active={combined.handBrake} aria-label="Handbrake" onpointerdown={(event) => pointerDown(event, 'handBrake')} onpointerup={pointerUp} onpointercancel={pointerUp} onlostpointercapture={pointerUp} oncontextmenu={preventContextMenu}><strong>P</strong><small>Brake</small></button>
    </div>
  </div>
</section>
