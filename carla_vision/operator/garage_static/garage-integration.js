"use strict";

(() => {
  const autonomousModes = new Set(["behavior", "imitation", "voxel"]);
  let checkpointMode = "";

  function element(id) {
    return document.getElementById(id);
  }

  function selectedMode() {
    return element("drive-garage-control-mode")?.value || "manual";
  }

  function isAutonomousMode(mode = selectedMode()) {
    return autonomousModes.has(mode);
  }

  function createOption(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }

  function populateCounts(select, values) {
    if (!select || select.dataset.garagePopulated === "true") return;
    select.replaceChildren();
    for (const [value, label] of values) select.append(createOption(String(value), label));
    select.value = "0";
    select.dataset.garagePopulated = "true";
  }

  function modeLabel(mode) {
    return {
      manual: "MANUAL",
      behavior: "BEHAVIOR AGENT",
      imitation: "IMITATION",
      voxel: "VOXEL PLANNER",
    }[mode] || String(mode || "manual").toUpperCase();
  }

  function installModeControls() {
    const form = element("drive-start-form");
    if (!form || element("drive-garage-control-mode")) return;
    const carGroup = form.querySelector("fieldset.garage-group");
    const fieldset = document.createElement("fieldset");
    fieldset.className = "garage-group garage-drive-mode";
    fieldset.innerHTML = `
      <legend>Driving mode</legend>
      <div class="fields two">
        <label>
          Control
          <select id="drive-garage-control-mode" required>
            <option value="manual">Manual</option>
          </select>
        </label>
        <label id="drive-behavior-wrap">
          Driving style
          <select id="drive-behavior">
            <option value="cautious">Cautious</option>
            <option value="normal" selected>Normal</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </label>
        <label id="drive-policy-checkpoint-wrap" class="span-2" hidden>
          Policy checkpoint
          <select id="drive-policy-checkpoint">
            <option value="">No compatible checkpoint found</option>
          </select>
          <small>Imitation and Voxel modes use the selected workspace checkpoint.</small>
        </label>
        <label id="drive-policy-device-wrap" hidden>
          Policy device
          <select id="drive-policy-device">
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
          </select>
        </label>
        <label id="drive-target-speed-wrap">
          Target speed
          <select id="drive-target-speed">
            <option value="25">25 km/h</option>
            <option value="35" selected>35 km/h</option>
            <option value="45">45 km/h</option>
          </select>
        </label>
        <label id="drive-voxel-readiness-wrap" class="span-2" hidden>
          Voxel readiness report <small>optional</small>
          <input id="drive-voxel-readiness" placeholder="runs/.../report.json">
        </label>
      </div>
      <label id="drive-autonomy-ack-wrap" class="garage-autonomy-ack" hidden>
        <input id="drive-autonomy-ack" type="checkbox">
        <span>
          <strong>Arm autonomous control</strong>
          <small>I understand this mode will actuate the CARLA ego vehicle. Emergency Brake remains independent.</small>
        </span>
      </label>
      <p id="drive-mode-note" class="garage-mode-note">Browser keyboard and touch controls own the vehicle.</p>
    `;
    if (carGroup?.nextSibling) form.insertBefore(fieldset, carGroup.nextSibling);
    else form.prepend(fieldset);

    const hudSession = element("drive-hud-session");
    if (hudSession && !element("drive-garage-mode-badge")) {
      const badge = document.createElement("span");
      badge.id = "drive-garage-mode-badge";
      badge.className = "drive-garage-mode-badge";
      badge.textContent = "MANUAL";
      hudSession.insertAdjacentElement("afterend", badge);
    }

    element("drive-garage-control-mode").addEventListener("change", () => {
      checkpointMode = "";
      syncModeUi(true);
    });
  }

  function syncModeOptions() {
    const select = element("drive-garage-control-mode");
    const catalog = state.drive?.catalog;
    if (!select || !catalog) return;
    const current = select.value || "manual";
    const modes = catalog.control_modes || [{ id: "manual", label: "Manual", available: true }];
    const signature = JSON.stringify(modes);
    if (select.dataset.catalogSignature === signature) return;
    select.replaceChildren();
    for (const mode of modes) {
      const option = createOption(
        mode.id,
        mode.available ? mode.label : `${mode.label} · unavailable`,
      );
      option.disabled = !mode.available;
      select.append(option);
    }
    select.dataset.catalogSignature = signature;
    if ([...select.options].some((option) => option.value === current && !option.disabled)) {
      select.value = current;
    } else {
      select.value = "manual";
    }
  }

  function syncCheckpointOptions(mode) {
    const select = element("drive-policy-checkpoint");
    if (!select || checkpointMode === mode) return;
    const weights = state.drive?.catalog?.policy_checkpoints || state.catalog?.weights || [];
    const filtered = weights.filter((path) => {
      const value = String(path).toLowerCase();
      if (mode === "imitation") return value.includes("imitation") || value.includes("behavior");
      if (mode === "voxel") return value.includes("voxel") || value.includes("flow");
      return false;
    });
    const choices = filtered.length ? filtered : weights;
    const previous = select.value;
    select.replaceChildren();
    if (!choices.length) select.append(createOption("", "No checkpoint found in workspace"));
    for (const path of choices) select.append(createOption(path, path));
    if ([...select.options].some((option) => option.value === previous)) select.value = previous;
    checkpointMode = mode;
  }

  function syncPopulationControls() {
    const active = driveIsActive();
    const capabilities = state.drive?.catalog?.capabilities || {};
    const worker = typeof driveWorldCapabilities === "function"
      ? driveWorldCapabilities()
      : { nativeWorker: false, traffic: false, walkers: false };
    const trafficAvailable = worker.traffic || capabilities.garage_traffic_population;
    const walkersAvailable = worker.walkers || capabilities.garage_walker_population;
    const traffic = element("drive-traffic-choice");
    const walkers = element("drive-walkers-choice");
    const trafficField = element("drive-traffic-field");
    const walkersField = element("drive-walkers-field");
    populateCounts(traffic, [
      [0, "None"],
      [15, "Light · 15"],
      [35, "Medium · 35"],
      [60, "Heavy · 60"],
    ]);
    populateCounts(walkers, [
      [0, "None"],
      [10, "Light · 10"],
      [25, "Medium · 25"],
      [40, "Busy · 40"],
    ]);
    if (trafficField) trafficField.hidden = !trafficAvailable;
    if (walkersField) walkersField.hidden = !walkersAvailable;
    if (traffic) traffic.disabled = active || !trafficAvailable;
    if (walkers) walkers.disabled = active || !walkersAvailable;
    const note = document.querySelector(".garage-worker-note");
    if (note && worker.nativeWorker) {
      note.textContent = "World Worker is connected. Maps, routes, traffic and pedestrians are owned by this drive session and cleaned up when it ends.";
    } else if (
      note &&
      (capabilities.garage_traffic_population || capabilities.garage_walker_population)
    ) {
      note.textContent = "Local CARLA PythonAPI is available. Garage traffic and pedestrians are owned by this drive session and cleaned up when it ends.";
    }
  }

  function syncModeUi(force = false) {
    syncModeOptions();
    const mode = selectedMode();
    const autonomous = isAutonomousMode(mode);
    const modelMode = mode === "imitation" || mode === "voxel";
    if (autonomous && typeof setDriveInitialControlMode === "function") {
      setDriveInitialControlMode("manual");
    }
    if (modelMode) syncCheckpointOptions(mode);

    const active = driveIsActive();
    const control = element("drive-garage-control-mode");
    if (control) control.disabled = active;
    element("drive-behavior-wrap").hidden = !["behavior", "voxel"].includes(mode);
    element("drive-policy-checkpoint-wrap").hidden = !modelMode;
    element("drive-policy-device-wrap").hidden = !modelMode;
    element("drive-target-speed-wrap").hidden = !["behavior", "voxel"].includes(mode);
    element("drive-voxel-readiness-wrap").hidden = mode !== "voxel";
    element("drive-autonomy-ack-wrap").hidden = !autonomous;

    for (const id of [
      "drive-behavior",
      "drive-policy-checkpoint",
      "drive-policy-device",
      "drive-target-speed",
      "drive-voxel-readiness",
      "drive-autonomy-ack",
    ]) {
      const node = element(id);
      if (node) node.disabled = active;
    }

    const note = element("drive-mode-note");
    if (note) {
      note.textContent = {
        manual: "Browser keyboard and touch controls own the vehicle.",
        behavior: "BehaviorAgent owns steering, throttle and brake. Browser input is locked out.",
        imitation: "The RGB + speed imitation policy owns steering, throttle and brake. Failures apply full brake.",
        voxel: "BehaviorAgent owns longitudinal control; the supervised voxel planner owns steering. Rejected predictions apply full brake.",
      }[mode];
    }

    const detector = element("drive-detector-enabled");
    if (detector && autonomous) {
      detector.checked = false;
      detector.disabled = true;
      if (typeof setDriveView === "function") setDriveView("raw");
    } else if (detector && !active) {
      detector.disabled = false;
    }

    document.body.classList.toggle("garage-autonomy-selected", autonomous);
    document.body.classList.toggle("garage-autonomous", autonomous && driveIsRunning());
    const badge = element("drive-garage-mode-badge");
    const liveMode = state.drive?.session?.garage_mode || mode;
    if (badge) badge.textContent = modeLabel(liveMode);

    const focus = element("drive-focus");
    if (focus && autonomous) focus.disabled = true;
    const deadman = element("drive-deadman");
    const sessionAutonomy = state.drive?.session?.autonomy;
    if (deadman && autonomous && driveIsRunning()) {
      if (sessionAutonomy?.failsafes > 0 && state.drive.session.deadman_active) {
        deadman.textContent = "SAFE BRAKE";
        deadman.className = "status-pill bad";
      } else {
        deadman.textContent = "AUTONOMY ACTIVE";
        deadman.className = "status-pill ok";
      }
    }
    if (force) syncPopulationControls();
  }

  function modelModeCheckpoint(mode) {
    if (!["imitation", "voxel"].includes(mode)) return "";
    return element("drive-policy-checkpoint").value;
  }

  function fullStartPayload() {
    const mode = selectedMode();
    const autonomous = isAutonomousMode(mode);
    const base = driveStartPayload();
    const workerOwnsWorld = driveWorldCapabilities().nativeWorker;
    return {
      ...base,
      traffic_count: workerOwnsWorld ? base.traffic_count : 0,
      walker_count: workerOwnsWorld ? base.walker_count : 0,
      initial_control_mode: autonomous ? "manual" : base.initial_control_mode,
      detector_enabled: autonomous ? false : checked("drive-detector-enabled"),
      weights: autonomous ? "" : element("drive-weights").value,
      control_mode: mode,
      behavior: element("drive-behavior").value,
      acknowledge_autonomy: autonomous && checked("drive-autonomy-ack"),
      traffic_vehicles: workerOwnsWorld
        ? 0
        : Number(element("drive-traffic-choice")?.value || 0),
      walkers: workerOwnsWorld
        ? 0
        : Number(element("drive-walkers-choice")?.value || 0),
      tm_port: 8000,
      target_speed_kmh: Number(element("drive-target-speed").value || 35),
      policy_checkpoint: modelModeCheckpoint(mode),
      policy_device: element("drive-policy-device").value,
      voxel_readiness_report:
        mode === "voxel" ? element("drive-voxel-readiness").value.trim() : "",
    };
  }

  async function startGarageDrive(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
    const button = event.submitter || element("drive-start");
    button.disabled = true;
    clearDriveKeys();
    try {
      const payload = await request("/api/drive/start", {
        method: "POST",
        body: JSON.stringify(fullStartPayload()),
      });
      state.drive.session = payload.state || payload;
      state.drive.sequence = 0;
      state.drive.lastTerminalSession = null;
      state.drive.inputFocused = false;
      renderDriveState();
      syncModeUi(true);
      const mode = state.drive.session.garage_mode || selectedMode();
      showToast(
        mode === "manual"
          ? "Drive is starting. Click the camera when it appears to take control."
          : `${modeLabel(mode)} is starting. Browser driving input is locked; Emergency Brake remains available.`,
      );
    } catch (error) {
      showToast(error.message, true);
      await refreshDriveState();
    } finally {
      renderDriveState();
      syncModeUi(true);
    }
  }

  function blockManualControlInAutonomy(event) {
    const mode = state.drive?.session?.garage_mode || selectedMode();
    if (!isAutonomousMode(mode) || !driveIsRunning()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    state.drive.inputFocused = false;
    clearDriveKeys();
    element("drive-viewport")?.blur();
  }

  function savedRunPath() {
    const session = state.drive?.session || {};
    if (session.status !== "success") return "";
    return typeof session.output_path === "string" ? session.output_path : "";
  }

  async function refreshSavedRun(path) {
    await refreshBootstrap();
    const exists = (state.catalog?.research_objects || []).some((row) => row.path === path);
    if (!exists) throw new Error(`Saved run ${path} is not present in the research catalog.`);
  }

  async function inspectSavedRun(button) {
    const path = savedRunPath();
    if (!path) return showToast("Finish and save a drive before opening it in Research.", true);
    button.disabled = true;
    try {
      await refreshSavedRun(path);
      activateTab("tools");
      activateResearchTool("evidence");
      await selectEvidence(path);
      element("panel-evidence")?.scrollIntoView({ behavior: "smooth", block: "start" });
      showToast("Opened this drive in Saved results.");
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function verifySavedRun(button) {
    const path = savedRunPath();
    if (!path) return showToast("Finish and save a drive before verifying it.", true);
    button.disabled = true;
    try {
      await refreshSavedRun(path);
      await startJob("verify", {
        paths: [path],
        allow_non_success: false,
        require_clean_git: false,
      });
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function analyzeSavedRun(button) {
    const path = savedRunPath();
    if (!path) return showToast("Finish and save a drive before analyzing it.", true);
    button.disabled = true;
    try {
      await refreshSavedRun(path);
      await startJob("analyze", {
        source_run: path,
        run_id: generatedId("analysis-drive"),
      });
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function installSavedRunActions() {
    const banner = element("drive-result-banner");
    if (!banner || element("drive-research-actions")) return;
    const panel = document.createElement("div");
    panel.id = "drive-research-actions";
    panel.className = "drive-research-actions";
    panel.innerHTML = `
      <div class="drive-research-copy">
        <strong>Continue with this run</strong>
        <span>The saved run is already a research object; no path re-entry is needed.</span>
      </div>
      <div class="drive-research-buttons"></div>
    `;
    const buttons = panel.querySelector(".drive-research-buttons");
    for (const [id, label, handler] of [
      ["drive-inspect-run", "Inspect", inspectSavedRun],
      ["drive-analyze-run", "Analyze", analyzeSavedRun],
      ["drive-verify-run", "Verify", verifySavedRun],
    ]) {
      const button = document.createElement("button");
      button.id = id;
      button.type = "button";
      button.className = "secondary";
      button.textContent = label;
      button.addEventListener("click", () => void handler(button));
      buttons.append(button);
    }
    banner.append(panel);
  }

  function installResearchShelf() {
    const form = element("drive-start-form");
    if (!form || element("garage-research-shelf")) return;
    const shelf = document.createElement("section");
    shelf.id = "garage-research-shelf";
    shelf.className = "garage-research-shelf";
    shelf.innerHTML = `
      <div>
        <strong>Research</strong>
        <span>Open the existing project tools without leaving the operator.</span>
      </div>
      <div class="garage-research-shortcuts">
        <button type="button" data-garage-tool="capture">Capture</button>
        <button type="button" data-garage-tool="situations">Scene</button>
        <button type="button" data-garage-tool="workflows">Train / Replay</button>
        <button type="button" data-garage-tool="evidence">Results</button>
        <button type="button" data-garage-tool="sessions">Activity</button>
      </div>
    `;
    for (const button of shelf.querySelectorAll("[data-garage-tool]")) {
      button.addEventListener("click", () => {
        activateTab("tools");
        activateResearchTool(button.dataset.garageTool);
      });
    }
    form.append(shelf);
  }

  function install() {
    installModeControls();
    installSavedRunActions();
    installResearchShelf();
    const form = element("drive-start-form");
    form?.addEventListener("submit", startGarageDrive, { capture: true });
    for (const target of [element("drive-viewport"), element("drive-focus")]) {
      target?.addEventListener("click", blockManualControlInAutonomy, { capture: true });
      target?.addEventListener("pointerdown", blockManualControlInAutonomy, { capture: true });
      target?.addEventListener("focus", blockManualControlInAutonomy, { capture: true });
    }
    window.setInterval(() => {
      syncPopulationControls();
      syncModeUi();
    }, 250);
  }

  window.carlaGarageManualControlBlocked = () => {
    const mode = state.drive?.session?.garage_mode || selectedMode();
    return isAutonomousMode(mode) && driveIsRunning();
  };

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
