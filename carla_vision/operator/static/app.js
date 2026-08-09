"use strict";

const state = {
  token: "",
  catalog: null,
  jobs: [],
  selectedJobId: null,
  selectedStream: "stdout",
  selectedEvidencePath: null,
  selectedEvidence: null,
  toastTimer: null,
  drive: {
    catalog: null,
    session: { status: "idle" },
    view: "raw",
    inputFocused: false,
    keys: {
      forward: false,
      brake: false,
      left: false,
      right: false,
      handBrake: false,
      reverseModifier: false,
    },
    sequence: 0,
    controlInFlight: false,
    controlFailed: false,
    stateInFlight: false,
    frameLoading: false,
    frameTimer: null,
    pollFailed: false,
    lastSafetyStopAt: 0,
    lastTerminalSession: null,
  },
};

const $ = (id) => document.getElementById(id);

function nowToken() {
  const date = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return [
    date.getUTCFullYear(),
    pad(date.getUTCMonth() + 1),
    pad(date.getUTCDate()),
    "t",
    pad(date.getUTCHours()),
    pad(date.getUTCMinutes()),
    pad(date.getUTCSeconds()),
    "z",
  ].join("");
}

function generatedId(prefix) {
  return `${prefix}-${nowToken()}`;
}

function number(id) {
  return Number($(id).value);
}

function checked(id) {
  return $(id).checked;
}

function showToast(message, isError = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 4200);
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (options.method && options.method !== "GET") {
    headers["X-Operator-Token"] = state.token;
  }
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || `Request failed: ${response.status}`);
  }
  return payload;
}

function setOptions(id, values, placeholder = "No compatible item found", preferred = "") {
  const select = $(id);
  const previous = select.value;
  select.replaceChildren();
  if (!values.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = placeholder;
    select.append(option);
    return;
  }
  for (const value of values) {
    const option = document.createElement("option");
    option.value = typeof value === "string" ? value : value.value;
    option.textContent = typeof value === "string" ? value : value.label;
    select.append(option);
  }
  const candidates = [previous, preferred].filter(Boolean);
  for (const candidate of candidates) {
    if ([...select.options].some((option) => option.value === candidate)) {
      select.value = candidate;
      break;
    }
  }
}

function updateRange(inputId, outputId, digits = 0) {
  const render = () => {
    $(outputId).value = Number($(inputId).value).toFixed(digits);
  };
  $(inputId).addEventListener("input", render);
  render();
}

function selectPreferredWeight() {
  const detector = $("live-detector").value;
  const weights = state.catalog?.weights || [];
  const preferred =
    weights.find((value) => value.toLowerCase().includes(detector === "rtdetr" ? "rtdetr" : "yolo")) ||
    weights[0] ||
    "";
  setOptions("live-weights", weights, "Add a .pt weight file", preferred);
  $("live-confidence").value = detector === "rtdetr" ? "0.2" : "0.05";
  $("live-confidence").dispatchEvent(new Event("input"));
}

function populateCatalog(catalog, weatherPresets, propPresets) {
  state.catalog = catalog;
  const capabilities = catalog.capabilities;
  const carla = $("carla-status");
  carla.textContent = capabilities.carla_tcp_reachable
    ? "CARLA · reachable"
    : "CARLA · offline";
  carla.className = `status-pill ${capabilities.carla_tcp_reachable ? "ok" : "bad"}`;
  const pythonApi = $("pythonapi-status");
  pythonApi.textContent = capabilities.native_pythonapi_importable
    ? "PythonAPI · ready"
    : "PythonAPI · missing";
  pythonApi.className = `status-pill ${
    capabilities.native_pythonapi_importable ? "ok" : "bad"
  }`;

  const defaults = catalog.defaults;
  $("live-host").value = defaults.carla_host;
  $("live-port").value = defaults.carla_port;
  $("live-vehicle").value = defaults.vehicle_id;
  $("live-camera").value = defaults.camera_id;
  $("live-map").value = defaults.map;

  setOptions("situation-weather", weatherPresets);
  setOptions("situation-props", propPresets);
  setOptions("plan-suite", catalog.scenario_suites, "Save a situation first");
  setOptions(
    "plan-split",
    catalog.split_plans,
    "No split plan found",
    "configs/scenarios/split_plan_operator_development_v1.json",
  );
  setOptions("native-plan", catalog.scenario_plans, "Create a scenario plan first");
  setOptions("qa-dataset", catalog.datasets, "No manifest-backed dataset found");
  setOptions("matrix-config", catalog.shadow_configs, "No matrix config found");
  setOptions("replay-config", catalog.replay_configs, "No replay config found");
  setOptions("replay-dataset", catalog.datasets, "No manifest-backed dataset found");
  setOptions("replay-evaluation", catalog.evaluation_configs, "No evaluation config found");
  setOptions("train-config", catalog.training_configs, "No training config found");
  setOptions("train-dataset", catalog.datasets, "No manifest-backed dataset found");
  setOptions("train-weights", catalog.weights, "No .pt weights found");
  setOptions("analysis-source", catalog.runtime_runs, "No recorded runtime found");
  setOptions(
    "verify-path",
    catalog.research_objects.map((row) => ({
      value: row.path,
      label: `${row.id} · ${row.status}`,
    })),
    "No research object found",
  );
  selectPreferredWeight();
  selectPreferredDriveWeight();
  populateEvidenceFilters();
  renderEvidenceList();
  if (
    state.selectedEvidencePath &&
    catalog.research_objects.some((row) => row.path === state.selectedEvidencePath)
  ) {
    selectEvidence(state.selectedEvidencePath);
  }
}

function populateEvidenceFilters() {
  const rows = state.catalog?.research_objects || [];
  const rootCounts = state.catalog?.research_object_counts?.by_root || {};
  const statusCounts = state.catalog?.research_object_counts?.by_status || {};
  const roots = Object.keys(rootCounts);
  const statuses = Object.keys(statusCounts);
  setOptions(
    "evidence-root-filter",
    [
      {
        value: "",
        label: `All roots · ${rows.length}`,
      },
      ...roots.map((root) => ({
        value: root,
        label: `${root} · ${rootCounts[root] || 0}`,
      })),
    ],
  );
  setOptions(
    "evidence-status-filter",
    [
      {
        value: "",
        label: `All statuses · ${rows.length}`,
      },
      ...statuses.map((status) => ({
        value: status,
        label: `${status} · ${statusCounts[status] || 0}`,
      })),
    ],
  );
  $("evidence-count").textContent = rows.length;
  $("evidence-root-counts").textContent =
    roots.map((root) => `${root} ${rootCounts[root] || 0}`).join(" · ") ||
    "No research objects discovered.";
}

function activateTab(name) {
  if (name !== "drive" && state.drive.inputFocused) {
    releaseDriveControl("Drive console hidden");
  }
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  }
}

async function startJob(kind, parameters) {
  const payload = {
    schema_version: "1.0",
    kind,
    parameters,
  };
  const job = await request("/api/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.selectedJobId = job.job_id;
  showToast(`${job.title} queued.`);
  await refreshJobs();
  activateTab("sessions");
}

function driveIsRunning() {
  return state.drive.session?.status === "running";
}

function driveIsActive() {
  return ["starting", "running", "stopping"].includes(state.drive.session?.status);
}

function driveEmergencyLatched() {
  return state.drive.session?.control_source === "emergency_stop";
}

function driveSessionId() {
  return state.drive.session?.session_id || "";
}

function drivePercent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : "0%";
}

function driveElapsed(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function driveCapabilityValue(capabilities, keys) {
  for (const key of keys) {
    if (!(key in capabilities)) continue;
    const value = capabilities[key];
    if (typeof value === "object" && value !== null) {
      if ("available" in value) return Boolean(value.available);
      if ("enabled" in value) return Boolean(value.enabled);
      if ("supported" in value) return Boolean(value.supported);
    }
    return Boolean(value);
  }
  return false;
}

function populateDriveColors() {
  const vehicleId = $("drive-vehicle").value;
  const vehicle = (state.drive.catalog?.vehicles || []).find((row) => row.id === vehicleId);
  const colors = (vehicle?.colors || []).map((color) =>
    typeof color === "string" ? { value: color, label: color } : color,
  );
  setOptions("drive-color", colors, "Blueprint default");
  $("drive-color").disabled = !colors.length || driveIsActive();
}

function selectPreferredDriveWeight() {
  const detector = $("drive-detector").value;
  const weights = state.catalog?.weights || [];
  const preferred =
    weights.find((value) =>
      value.toLowerCase().includes(detector === "rtdetr" ? "rtdetr" : "yolo"),
    ) ||
    weights[0] ||
    "";
  setOptions("drive-weights", weights, "Add a compatible .pt weight file", preferred);
  $("drive-confidence").value = detector === "rtdetr" ? "0.55" : "0.35";
  $("drive-confidence").dispatchEvent(new Event("input"));
  updateDriveConfigAvailability();
}

function renderDriveCapabilities() {
  const catalog = state.drive.catalog;
  if (!catalog) return;
  $("drive-server-connection").textContent = catalog.connected ? "reachable" : "offline";
  $("drive-server-connection").className = catalog.connected ? "fact-ok" : "fact-bad";
  $("drive-server-version").textContent = catalog.server_version || "unknown";
  $("drive-server-map").textContent = catalog.map || "unknown";
  $("drive-spawn-count").textContent = Number.isFinite(Number(catalog.spawn_count))
    ? String(catalog.spawn_count)
    : "unknown";

  const capabilities = catalog.capabilities || {};
  const rows = [
    ["Map reload", ["map_reload", "world_reload", "native_world_reload"]],
    ["Random route", ["random_route", "route_planning", "native_route_planning"]],
    ["Traffic population", ["traffic", "traffic_population", "spawn_traffic"]],
    ["Walker population", ["walkers", "walker_population", "spawn_walkers"]],
    ["Simulator autopilot", ["autopilot", "traffic_manager", "native_autopilot"]],
  ];
  const list = $("drive-capability-list");
  list.replaceChildren();
  for (const [label, keys] of rows) {
    const available = driveCapabilityValue(capabilities, keys);
    const row = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = label;
    const status = document.createElement("strong");
    status.className = available ? "capability-ready" : "capability-unavailable";
    status.textContent = available ? "Available via native workflow" : "Native worker unavailable";
    row.append(name, status);
    list.append(row);
  }
}

function populateDriveCatalog(catalog) {
  state.drive.catalog = catalog;
  if (catalog.host) $("drive-host").value = catalog.host;
  if (catalog.port) $("drive-port").value = catalog.port;
  const vehicles = (catalog.vehicles || []).map((vehicle) => ({
    value: vehicle.id,
    label: vehicle.label || vehicle.id,
  }));
  const preferredVehicle =
    vehicles.find((vehicle) => vehicle.value === "vehicle.tesla.model3")?.value ||
    vehicles[0]?.value ||
    "";
  setOptions("drive-vehicle", vehicles, "No compatible vehicle blueprint found", preferredVehicle);
  const weather = (catalog.weather_presets || []).map((preset) => ({
    value: preset.id,
    label: preset.label || preset.id,
  }));
  const preferredWeather =
    weather.find((preset) => preset.value.toLowerCase().includes("clear"))?.value ||
    weather[0]?.value ||
    "";
  setOptions("drive-weather", weather, "No weather presets reported", preferredWeather);
  const props = (catalog.prop_presets || []).map((preset) => ({
    value: preset.id,
    label: preset.label || preset.id,
  }));
  const preferredProps =
    props.find((preset) => ["none", "keep"].includes(preset.value))?.value ||
    props[0]?.value ||
    "";
  setOptions("drive-props", props, "No scene prop presets reported", preferredProps);
  populateDriveColors();
  renderDriveCapabilities();
  updateDriveConfigAvailability();
}

async function refreshDriveCatalog() {
  const catalog = await request("/api/drive/catalog");
  populateDriveCatalog(catalog.catalog || catalog);
}

function updateDriveModelToggle() {
  const enabled = checked("drive-detector-enabled");
  if (!enabled && state.drive.view === "overlay") {
    setDriveView("raw");
  }
  updateDriveConfigAvailability();
}

function updateDriveConfigAvailability() {
  const active = driveIsActive();
  const detectorEnabled = checked("drive-detector-enabled");
  const lockIds = [
    "drive-run-id",
    "drive-host",
    "drive-port",
    "drive-vehicle",
    "drive-seed",
    "drive-props",
    "drive-resolution",
    "drive-camera-fps",
    "drive-camera-fov",
    "drive-detector-enabled",
    "drive-record-video",
    "drive-spectator-follow",
  ];
  for (const id of lockIds) $(id).disabled = active;
  $("drive-color").disabled = active || $("drive-color").options.length === 1 && !$("drive-color").value;
  for (const id of [
    "drive-detector",
    "drive-weights",
    "drive-device",
    "drive-image-size",
    "drive-confidence",
  ]) {
    $(id).disabled = active || !detectorEnabled;
  }
  $("drive-view-overlay").disabled = !detectorEnabled;
}

function driveDetectorLabel(detector) {
  if (!detector) return "off";
  if (typeof detector === "string") return detector;
  if (detector.enabled === false) return "off";
  return detector.label || detector.name || detector.detector || detector.status || "enabled";
}

function renderDriveState() {
  const session = state.drive.session || { status: "idle" };
  const statusName = session.status || "idle";
  const status = $("drive-status");
  status.textContent = statusName;
  status.className = `status-pill ${statusClass(statusName)}`;
  const telemetry = session.telemetry || {};
  const speed = Number(telemetry.speed_mps ?? telemetry.speed ?? 0);
  $("drive-speed").textContent = Number.isFinite(speed) ? speed.toFixed(1) : "0.0";
  const gear = telemetry.gear;
  $("drive-gear").textContent = gear === -1 ? "R" : gear === 0 || gear == null ? "N" : String(gear);
  $("drive-elapsed").textContent = driveElapsed(session.elapsed_seconds);
  const age = !driveIsRunning() || session.input_age_seconds == null
    ? Number.NaN
    : Number(session.input_age_seconds);
  $("drive-input-age").textContent = Number.isFinite(age) ? `${age.toFixed(2)}s` : "—";
  $("drive-telemetry-throttle").textContent = drivePercent(telemetry.throttle);
  $("drive-telemetry-steer").textContent = drivePercent(telemetry.steer);
  $("drive-telemetry-brake").textContent = drivePercent(telemetry.brake);
  const released = ["success", "failed"].includes(statusName) ? " · released" : "";
  $("drive-vehicle-id").textContent = session.vehicle_id == null
    ? "—"
    : `${session.vehicle_id}${released}`;
  $("drive-camera-id").textContent = session.camera_id == null
    ? "—"
    : `${session.camera_id}${released}`;
  $("drive-detector-state").textContent = driveDetectorLabel(session.detector);
  $("drive-hud-session").textContent = driveSessionId()
    ? `SESSION ${driveSessionId()}`
    : statusName.toUpperCase();

  const recording = Boolean(
    typeof session.recording === "object" ? session.recording.active : session.recording,
  );
  const recordingBadge = $("drive-hud-recording");
  recordingBadge.textContent = recording ? "● REC" : "REC OFF";
  recordingBadge.classList.toggle("active", recording);

  const deadman = $("drive-deadman");
  if (driveIsRunning() && session.deadman_active) {
    deadman.textContent = "FULL BRAKE";
    deadman.className = "status-pill bad";
  } else if (driveIsRunning() && state.drive.inputFocused && !driveEmergencyLatched()) {
    deadman.textContent = "heartbeat live";
    deadman.className = "status-pill ok";
  } else if (driveIsRunning()) {
    deadman.textContent = "awaiting focus";
    deadman.className = "status-pill pending";
  } else {
    deadman.textContent = "inactive";
    deadman.className = "status-pill neutral";
  }

  $("drive-start").disabled =
    driveIsActive() ||
    !state.drive.catalog?.connected ||
    !$("drive-vehicle").value;
  $("drive-stop").disabled = !driveIsActive();
  $("drive-emergency").disabled = !driveIsRunning();
  $("drive-focus").disabled = !driveIsRunning() || driveEmergencyLatched();
  $("drive-weather").disabled = !(state.drive.catalog?.weather_presets || []).length;
  $("drive-props").disabled =
    driveIsActive() || !(state.drive.catalog?.prop_presets || []).length;
  $("drive-error").textContent = session.error || "";
  $("drive-output-path").textContent = session.output_path
    ? `Retained output: ${session.output_path}`
    : recording
      ? "Recording is active; output will be finalized on Stop & Save."
      : "No retained output yet.";

  const sequence =
    state.drive.view === "overlay"
      ? session.overlay_frame_sequence
      : session.raw_frame_sequence;
  $("drive-frame-state").textContent = driveIsRunning()
    ? `${state.drive.view === "overlay" ? "Model" : "Raw"} view · frame ${sequence ?? "—"}`
    : statusName === "starting"
      ? "Starting camera stream…"
      : statusName === "stopping"
        ? "Finalizing retained artifacts…"
        : "Waiting for a session";

  updateDriveConfigAvailability();
  if (driveIsRunning()) {
    scheduleDriveFrame(0);
  } else {
    state.drive.frameLoading = false;
    window.clearTimeout(state.drive.frameTimer);
  }

  if (["success", "failed"].includes(statusName) && driveSessionId()) {
    if (state.drive.lastTerminalSession !== driveSessionId()) {
      state.drive.lastTerminalSession = driveSessionId();
      releaseDriveControl("Session finished", false);
      $("drive-run-id").value = generatedId("drive");
      showToast(
        statusName === "success" ? "Drive stopped and retained artifacts finalized." : session.error || "Drive failed.",
        statusName === "failed",
      );
    }
  }
}

async function refreshDriveState() {
  if (state.drive.stateInFlight) return;
  state.drive.stateInFlight = true;
  try {
    const payload = await request("/api/drive/state");
    state.drive.session = payload.state || payload;
    state.drive.pollFailed = false;
    renderDriveState();
  } catch (error) {
    if (!state.drive.pollFailed) {
      state.drive.pollFailed = true;
      showToast(`Drive state unavailable: ${error.message}`, true);
    }
  } finally {
    state.drive.stateInFlight = false;
  }
}

function setDriveView(view) {
  state.drive.view = view === "overlay" ? "overlay" : "raw";
  for (const candidate of ["raw", "overlay"]) {
    const button = $(`drive-view-${candidate}`);
    const active = candidate === state.drive.view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  state.drive.frameLoading = false;
  window.clearTimeout(state.drive.frameTimer);
  renderDriveState();
}

function scheduleDriveFrame(delay = 80) {
  if (!driveIsRunning()) return;
  window.clearTimeout(state.drive.frameTimer);
  state.drive.frameTimer = window.setTimeout(refreshDriveFrame, delay);
}

function refreshDriveFrame() {
  if (
    !driveIsRunning() ||
    state.drive.frameLoading ||
    !$("panel-drive").classList.contains("active")
  ) {
    scheduleDriveFrame(120);
    return;
  }
  state.drive.frameLoading = true;
  const requestedView = state.drive.view;
  const candidate = new Image();
  candidate.onload = () => {
    state.drive.frameLoading = false;
    if (driveIsRunning() && requestedView === state.drive.view) {
      const frame = $("drive-frame");
      frame.src = candidate.src;
      frame.style.display = "block";
      $("drive-empty").style.display = "none";
    }
    scheduleDriveFrame(65);
  };
  candidate.onerror = () => {
    state.drive.frameLoading = false;
    scheduleDriveFrame(220);
  };
  candidate.src = `/api/drive/frame.jpg?view=${encodeURIComponent(requestedView)}&t=${Date.now()}`;
}

function clearDriveKeys() {
  for (const key of Object.keys(state.drive.keys)) state.drive.keys[key] = false;
  renderDriveKeyState();
}

function currentDriveCommand() {
  if (!driveIsRunning() || !state.drive.inputFocused || driveEmergencyLatched()) {
    return {
      throttle: 0,
      steer: 0,
      brake: 1,
      hand_brake: false,
      reverse: false,
    };
  }
  const keys = state.drive.keys;
  const reverseRequested = keys.reverseModifier && keys.forward;
  const telemetry = state.drive.session?.telemetry || {};
  const speed = Math.abs(Number(telemetry.speed_mps ?? telemetry.speed ?? 0));
  const alreadyInReverse = Number(telemetry.gear) === -1;
  const reverse = reverseRequested && (alreadyInReverse || speed <= 0.5);
  let throttle = keys.forward ? (reverse ? 0.35 : 0.55) : 0;
  let brake = keys.brake ? 0.85 : 0;
  let handBrake = keys.handBrake;
  let steer = Number(keys.right) * 0.7 - Number(keys.left) * 0.7;
  if (reverseRequested && !reverse) {
    throttle = 0;
    brake = Math.max(brake, 0.65);
  }
  if (handBrake) {
    throttle = 0;
    brake = 1;
    steer = 0;
  }
  return {
    throttle,
    steer,
    brake,
    hand_brake: handBrake,
    reverse,
  };
}

function renderDriveKeyState() {
  const keys = state.drive.keys;
  $("drive-key-w").classList.toggle("active", keys.forward);
  $("drive-key-s").classList.toggle("active", keys.brake);
  $("drive-key-a").classList.toggle("active", keys.left);
  $("drive-key-d").classList.toggle("active", keys.right);
  const command = currentDriveCommand();
  $("drive-command-throttle").textContent = drivePercent(command.throttle);
  $("drive-command-steer").textContent = drivePercent(command.steer);
  $("drive-command-brake").textContent = drivePercent(command.brake);
  $("drive-viewport").classList.toggle("reversing", command.reverse);
}

async function sendDriveControl({ safety = false, keepalive = false } = {}) {
  const sessionId = driveSessionId();
  if (!sessionId) return;
  if (!safety && (!driveIsRunning() || !state.drive.inputFocused || state.drive.controlInFlight)) {
    return;
  }
  const command = safety
    ? { throttle: 0, steer: 0, brake: 1, hand_brake: false, reverse: false }
    : currentDriveCommand();
  const payload = {
    session_id: sessionId,
    sequence: ++state.drive.sequence,
    ...command,
  };
  if (!safety) state.drive.controlInFlight = true;
  try {
    await request("/api/drive/control", {
      method: "POST",
      body: JSON.stringify(payload),
      keepalive,
    });
    state.drive.controlFailed = false;
  } catch (error) {
    if (!safety && !state.drive.controlFailed) {
      state.drive.controlFailed = true;
      showToast(`Manual control failed: ${error.message}`, true);
    }
  } finally {
    if (!safety) state.drive.controlInFlight = false;
  }
}

function releaseDriveControl(reason = "Driving focus released", sendBrake = true) {
  state.drive.inputFocused = false;
  clearDriveKeys();
  $("drive-viewport").classList.remove("focused");
  $("drive-focus-shield").textContent = `${reason} · full brake requested`;
  $("drive-focus-shield").classList.remove("hidden");
  if (sendBrake && driveIsRunning()) {
    const now = Date.now();
    if (now - state.drive.lastSafetyStopAt > 80) {
      state.drive.lastSafetyStopAt = now;
      void sendDriveControl({ safety: true, keepalive: true });
    }
  }
  renderDriveState();
}

function focusDriveControl() {
  if (!driveIsRunning()) {
    showToast("Start a drive before focusing keyboard control.", true);
    return;
  }
  if (driveEmergencyLatched()) {
    showToast("Emergency stop is latched. Stop & Save, then start a new session.", true);
    return;
  }
  $("drive-viewport").focus({ preventScroll: true });
}

function driveKeyName(event) {
  const key = event.key.toLowerCase();
  if (key === "w" || key === "arrowup") return "forward";
  if (key === "s" || key === "arrowdown") return "brake";
  if (key === "a" || key === "arrowleft") return "left";
  if (key === "d" || key === "arrowright") return "right";
  if (event.code === "Space") return "handBrake";
  if (key === "shift") return "reverseModifier";
  return "";
}

function handleDriveKey(event, pressed) {
  if (!state.drive.inputFocused || !driveIsRunning()) return;
  const key = driveKeyName(event);
  if (!key) return;
  event.preventDefault();
  if (key === "reverseModifier" && state.drive.keys.forward) {
    state.drive.keys.forward = false;
  }
  state.drive.keys[key] = pressed;
  renderDriveKeyState();
  void sendDriveControl();
}

async function startDrive(event) {
  event.preventDefault();
  const button = event.submitter || $("drive-start");
  button.disabled = true;
  clearDriveKeys();
  try {
    const payload = await request("/api/drive/start", {
      method: "POST",
      body: JSON.stringify({
        run_id: $("drive-run-id").value,
        host: $("drive-host").value,
        port: number("drive-port"),
        vehicle_blueprint: $("drive-vehicle").value,
        color: $("drive-color").value,
        seed: number("drive-seed"),
        weather_preset: $("drive-weather").value,
        prop_preset: $("drive-props").value,
        detector_enabled: checked("drive-detector-enabled"),
        detector: $("drive-detector").value,
        weights: $("drive-weights").value,
        device: $("drive-device").value,
        image_size: number("drive-image-size"),
        confidence: number("drive-confidence"),
        resolution: $("drive-resolution").value,
        camera_fps: number("drive-camera-fps"),
        camera_fov: number("drive-camera-fov"),
        record_video: checked("drive-record-video"),
        spectator_follow: checked("drive-spectator-follow"),
      }),
    });
    state.drive.session = payload.state || payload;
    state.drive.sequence = 0;
    state.drive.lastTerminalSession = null;
    renderDriveState();
    showToast("Manual drive is starting. Focus the camera viewport when it is ready.");
  } catch (error) {
    showToast(error.message, true);
    await refreshDriveState();
  } finally {
    renderDriveState();
  }
}

async function stopDrive() {
  if (!driveSessionId()) return;
  releaseDriveControl("Stop & Save requested");
  $("drive-stop").disabled = true;
  try {
    const payload = await request("/api/drive/stop", {
      method: "POST",
      body: JSON.stringify({ session_id: driveSessionId() }),
    });
    state.drive.session = payload.state || payload;
    renderDriveState();
    showToast("Stopping drive and finalizing retained output…");
  } catch (error) {
    showToast(error.message, true);
    await refreshDriveState();
  }
}

async function emergencyStopDrive() {
  if (!driveSessionId()) return;
  releaseDriveControl("Emergency stop requested");
  try {
    const payload = await request("/api/drive/emergency-stop", {
      method: "POST",
      body: JSON.stringify({ session_id: driveSessionId() }),
    });
    if (payload.state || payload.status) state.drive.session = payload.state || payload;
    renderDriveState();
    showToast("Emergency stop is latched. Use Stop & Save before starting another drive.");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function changeDriveWeather() {
  if (!driveIsRunning()) return;
  if ($("drive-weather").value === "keep") {
    showToast("Keep current weather is a start-only option; the live world was not changed.");
    return;
  }
  try {
    await request("/api/drive/weather", {
      method: "POST",
      body: JSON.stringify({
        session_id: driveSessionId(),
        preset: $("drive-weather").value,
      }),
    });
    showToast("Weather update requested for the live world.");
  } catch (error) {
    showToast(error.message, true);
  }
}

function bindDriveConsole() {
  $("drive-run-id").value = generatedId("drive");
  updateRange("drive-confidence", "drive-confidence-value", 2);
  $("drive-start-form").addEventListener("submit", startDrive);
  $("drive-stop").addEventListener("click", stopDrive);
  $("drive-emergency").addEventListener("click", emergencyStopDrive);
  $("drive-focus").addEventListener("click", focusDriveControl);
  $("drive-view-raw").addEventListener("click", () => setDriveView("raw"));
  $("drive-view-overlay").addEventListener("click", () => setDriveView("overlay"));
  $("drive-vehicle").addEventListener("change", populateDriveColors);
  $("drive-detector").addEventListener("change", selectPreferredDriveWeight);
  $("drive-detector-enabled").addEventListener("change", updateDriveModelToggle);
  $("drive-weather").addEventListener("change", changeDriveWeather);

  const viewport = $("drive-viewport");
  viewport.addEventListener("click", focusDriveControl);
  viewport.addEventListener("focus", () => {
    if (!driveIsRunning()) return;
    state.drive.inputFocused = true;
    viewport.classList.add("focused");
    $("drive-focus-shield").classList.add("hidden");
    renderDriveState();
    void sendDriveControl();
  });
  viewport.addEventListener("blur", () => releaseDriveControl("Viewport focus lost"));
  viewport.addEventListener("keydown", (event) => handleDriveKey(event, true));
  viewport.addEventListener("keyup", (event) => handleDriveKey(event, false));
  window.addEventListener("keyup", (event) => handleDriveKey(event, false));
  window.addEventListener("blur", () => releaseDriveControl("Browser focus lost"));
  window.addEventListener("pagehide", () => releaseDriveControl("Page closing"));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) releaseDriveControl("Page hidden");
  });
  window.setInterval(() => void sendDriveControl(), 67);
  renderDriveKeyState();
  renderDriveState();
}

function bindLiveForm() {
  $("live-run-id").value = generatedId("ui-live");
  $("live-detector").addEventListener("change", selectPreferredWeight);
  $("live-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter;
    button.disabled = true;
    try {
      await startJob("live", {
        run_id: $("live-run-id").value,
        host: $("live-host").value,
        port: number("live-port"),
        vehicle_id: number("live-vehicle"),
        camera_id: number("live-camera"),
        resolution: $("live-resolution").value,
        camera_fps: number("live-camera-fps"),
        camera_fov: number("live-fov"),
        expected_map: $("live-map").value,
        detector: $("live-detector").value,
        weights: $("live-weights").value,
        model_package: "",
        device: $("live-device").value,
        image_size: number("live-image-size"),
        confidence: number("live-confidence"),
        control: $("live-control").value,
        cruise_speed: number("live-cruise-speed"),
        duration: number("live-duration"),
        max_stale_seconds: number("live-stale"),
        view: $("live-view").value,
        record_video: checked("live-video"),
        spectator_follow: checked("live-spectator-follow"),
        shadow_policy: $("live-policy").value,
        policy_options: {
          confidence: 0.35,
          close_bottom: number("live-close-bottom"),
          corridor_left: 0.3,
          corridor_right: 0.7,
          cruise_throttle: 0.15,
        },
        acknowledge_teacher_motion: checked("live-motion-ack"),
      });
      $("live-run-id").value = generatedId("ui-live");
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  });
}

function situationPayload() {
  return {
    situation_id: $("situation-id").value,
    map_name: $("situation-map").value,
    weather_preset: $("situation-weather").value,
    vehicle_count: number("vehicle-count"),
    walker_count: number("walker-count"),
    pedestrian_crossing_factor: number("crossing"),
    speed_difference_percent: number("speed-difference"),
    following_distance_metres: number("following-distance"),
    prop_preset: $("situation-props").value,
    ego_blueprint: $("situation-ego").value,
    ego_spawn_index: number("situation-spawn"),
    duration_seconds: number("situation-duration"),
    capture_fps: number("situation-capture-fps"),
    repetitions: number("situation-repetitions"),
    master_seed: number("situation-seed"),
    camera_width: number("situation-width"),
    camera_height: number("situation-height"),
    camera_fov: number("situation-fov"),
  };
}

async function refreshBootstrap() {
  const bootstrap = await request("/api/bootstrap");
  state.token = bootstrap.token;
  populateCatalog(
    bootstrap.catalog,
    bootstrap.weather_presets,
    bootstrap.prop_presets,
  );
  state.jobs = bootstrap.jobs;
  renderJobs();
}

function bindSituationForms() {
  $("plan-run-id").value = generatedId("ui-plan");
  $("situation-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter;
    button.disabled = true;
    const result = $("situation-result");
    try {
      const payload = await request("/api/situations", {
        method: "POST",
        body: JSON.stringify(situationPayload()),
      });
      result.textContent = `Saved ${payload.path}`;
      result.className = "inline-result ok";
      showToast("Situation recipe saved and validated.");
      await refreshBootstrap();
      $("plan-suite").value = payload.path;
      $("plan-run-id").value = `plan-${$("situation-id").value}-${nowToken()}`;
    } catch (error) {
      result.textContent = error.message;
      result.className = "inline-result";
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  });
  $("plan-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("scenario_plan", {
        suite: $("plan-suite").value,
        split_plan: $("plan-split").value,
        run_id: $("plan-run-id").value,
      });
      $("plan-run-id").value = generatedId("ui-plan");
    } catch (error) {
      showToast(error.message, true);
    }
  });
}

function bindWorkflowForms() {
  $("native-dataset-id").value = generatedId("ds-ui");
  $("qa-run-id").value = generatedId("dataset-qa-ui");
  $("matrix-id").value = generatedId("shadow-ui");
  $("replay-id").value = generatedId("replay-ui");
  $("train-run-id").value = generatedId("train-ui");
  $("analysis-run-id").value = generatedId("analysis-ui");

  $("native-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const dryRun = checked("native-dry-run");
      await startJob("native_capture", {
        scenario_plan: $("native-plan").value,
        dataset_id: $("native-dataset-id").value,
        host: state.catalog.defaults.carla_host,
        port: state.catalog.defaults.carla_port,
        partition: $("native-partition").value,
        max_episodes: number("native-max-episodes"),
        timeout: 30,
        sensor_timeout: 10,
        carla_python_api: $("native-pythonapi").value,
        dry_run: dryRun,
        acknowledge_exclusive_tick_owner: checked("native-ack"),
      });
      if (!dryRun) {
        $("native-dataset-id").value = generatedId("ds-ui");
      }
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("native-preflight").addEventListener("click", async () => {
    try {
      const confirmed = checked("native-ack");
      await startJob("native_preflight", {
        scenario_plan: $("native-plan").value,
        dataset_id: $("native-dataset-id").value,
        run_id: generatedId("native-preflight-ui"),
        host: state.catalog.defaults.carla_host,
        port: state.catalog.defaults.carla_port,
        partition: $("native-partition").value,
        max_episodes: number("native-max-episodes"),
        timeout: 3,
        carla_python_api: $("native-pythonapi").value,
        confirm_world_reload: confirmed,
        confirm_exclusive_tick_owner: confirmed,
      });
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("qa-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("dataset_qa", {
        dataset: $("qa-dataset").value,
        run_id: $("qa-run-id").value,
        montage_count: number("qa-montage-count"),
      });
      $("qa-run-id").value = generatedId("dataset-qa-ui");
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("matrix-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("shadow_matrix", {
        config: $("matrix-config").value,
        matrix_id: $("matrix-id").value,
        execute: checked("matrix-execute"),
        acknowledge_teacher_motion: checked("matrix-ack"),
      });
      $("matrix-id").value = generatedId("shadow-ui");
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("replay-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("replay", {
        config: $("replay-config").value,
        replay_id: $("replay-id").value,
        dataset: $("replay-dataset").value,
        evaluation_config: $("replay-evaluation").value,
        acknowledge_locked_test: checked("replay-locked"),
      });
      $("replay-id").value = generatedId("replay-ui");
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("train-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("train", {
        config: $("train-config").value,
        dataset: $("train-dataset").value,
        weights: $("train-weights").value,
        run_id: $("train-run-id").value,
        device: $("train-device").value,
        dry_run: checked("train-dry-run"),
        acknowledge_real_training: checked("train-ack"),
      });
      $("train-run-id").value = generatedId("train-ui");
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("analysis-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("analyze", {
        source_run: $("analysis-source").value,
        run_id: $("analysis-run-id").value,
      });
      $("analysis-run-id").value = generatedId("analysis-ui");
    } catch (error) {
      showToast(error.message, true);
    }
  });

  $("verify-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await startJob("verify", {
        paths: [$("verify-path").value],
        allow_non_success: checked("verify-non-success"),
        require_clean_git: checked("verify-clean-git"),
      });
    } catch (error) {
      showToast(error.message, true);
    }
  });
}

function statusClass(status) {
  if (status === "success") return "ok";
  if (["failed", "stopped"].includes(status)) return "bad";
  if (["queued", "starting", "running", "stopping", "finalizing"].includes(status)) {
    return "running";
  }
  return "neutral";
}

function isActive(status) {
  return ["queued", "starting", "running", "stopping", "finalizing"].includes(status);
}

function readableTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleTimeString();
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value < 0) return "size unavailable";
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = "B";
  for (const candidate of units) {
    amount /= 1024;
    unit = candidate;
    if (amount < 1024) break;
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${unit}`;
}

function shortHash(value) {
  return typeof value === "string" && value.length > 12 ? `${value.slice(0, 12)}…` : value || "—";
}

function artifactUrl(objectPath, artifactPath) {
  return `/api/artifact?object=${encodeURIComponent(objectPath)}&path=${encodeURIComponent(
    artifactPath,
  )}`;
}

function filteredEvidenceRows() {
  const search = $("evidence-search").value.trim().toLowerCase();
  const root = $("evidence-root-filter").value;
  const status = $("evidence-status-filter").value;
  return (state.catalog?.research_objects || []).filter((row) => {
    if (root && row.root_kind !== root) return false;
    if (status && row.status !== status) return false;
    if (!search) return true;
    const haystack = [
      row.id,
      row.path,
      row.root_kind,
      row.status,
      row.object_type,
      ...(row.roles || []),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(search);
  });
}

function renderEvidenceList() {
  const rows = filteredEvidenceRows();
  const body = $("evidence-body");
  body.replaceChildren();
  $("evidence-filter-count").textContent = `${rows.length} shown of ${
    state.catalog?.research_objects?.length || 0
  }`;
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "empty-state";
    cell.textContent = "No research objects match these filters.";
    row.append(cell);
    body.append(row);
    return;
  }
  for (const object of rows) {
    const row = document.createElement("tr");
    row.className = `session-row ${
      state.selectedEvidencePath === object.path ? "selected" : ""
    }`;
    row.addEventListener("click", () => selectEvidence(object.path));

    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = `status-pill ${statusClass(object.status)}`;
    status.textContent = object.status;
    status.title = "Manifest-declared status; verification has not been run by this view.";
    statusCell.append(status);

    const identityCell = document.createElement("td");
    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = object.id;
    const path = document.createElement("span");
    path.className = "session-id";
    path.textContent = object.path;
    identityCell.append(title, path);

    const typeCell = document.createElement("td");
    const root = document.createElement("span");
    root.className = "session-title";
    root.textContent = object.root_kind;
    const type = document.createElement("span");
    type.className = "session-id";
    type.textContent = object.object_type || "unknown";
    typeCell.append(root, type);

    const artifactCell = document.createElement("td");
    const count = document.createElement("span");
    count.className = "session-title";
    count.textContent = String(object.artifact_count || 0);
    const bytes = document.createElement("span");
    bytes.className = "session-id";
    bytes.textContent = formatBytes(object.declared_artifact_bytes);
    artifactCell.append(count, bytes);

    row.append(statusCell, identityCell, typeCell, artifactCell);
    body.append(row);
  }
}

function clearEvidencePreview() {
  const image = $("evidence-image-preview");
  const video = $("evidence-video-preview");
  image.removeAttribute("src");
  image.style.display = "none";
  video.pause();
  video.removeAttribute("src");
  video.load();
  video.style.display = "none";
  const empty = $("evidence-preview").querySelector(".empty-preview");
  empty.style.display = "block";
}

function previewEvidenceArtifact(objectPath, artifact) {
  clearEvidencePreview();
  const url = artifactUrl(objectPath, artifact.path);
  const empty = $("evidence-preview").querySelector(".empty-preview");
  if (artifact.preview_kind === "image") {
    const image = $("evidence-image-preview");
    image.src = url;
    image.alt = `${artifact.role}: ${artifact.path}`;
    image.style.display = "block";
    empty.style.display = "none";
  } else if (artifact.preview_kind === "video") {
    const video = $("evidence-video-preview");
    video.src = url;
    video.style.display = "block";
    empty.style.display = "none";
    video.load();
  }
}

function renderEvidenceArtifacts(payload) {
  const body = $("evidence-artifacts");
  body.replaceChildren();
  clearEvidencePreview();
  const artifacts = payload.artifacts || [];
  if (!artifacts.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.className = "empty-state";
    cell.textContent = "This manifest registers no artifacts.";
    row.append(cell);
    body.append(row);
    return;
  }
  let firstPreview = null;
  for (const artifact of artifacts) {
    const row = document.createElement("tr");

    const identityCell = document.createElement("td");
    const role = document.createElement("span");
    role.className = "session-title";
    role.textContent = artifact.role;
    const path = document.createElement("span");
    path.className = "session-id artifact-path";
    path.textContent = artifact.path;
    identityCell.append(role, path);

    const declarationCell = document.createElement("td");
    const declared = document.createElement("span");
    declared.className = "artifact-declaration";
    declared.textContent = `${artifact.mime_type} · ${formatBytes(artifact.size_bytes)}`;
    const hash = document.createElement("code");
    hash.className = "artifact-hash";
    hash.textContent = shortHash(artifact.sha256);
    hash.title = artifact.sha256 || "No declared SHA-256";
    declarationCell.append(declared, hash);
    if (Object.keys(artifact.metadata || {}).length) {
      const details = document.createElement("details");
      details.className = "artifact-metadata";
      const summary = document.createElement("summary");
      summary.textContent = "metadata";
      const content = document.createElement("pre");
      content.textContent = JSON.stringify(artifact.metadata, null, 2);
      details.append(summary, content);
      declarationCell.append(details);
    }

    const actionCell = document.createElement("td");
    if (artifact.downloadable) {
      const link = document.createElement("a");
      link.className = "artifact-action-link";
      link.href = artifactUrl(payload.object.path, artifact.path);
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = artifact.preview_kind === "download" ? "Download" : "Open";
      actionCell.append(link);
      if (["image", "video"].includes(artifact.preview_kind)) {
        const preview = document.createElement("button");
        preview.type = "button";
        preview.className = "row-action";
        preview.textContent = "Preview";
        preview.addEventListener("click", () =>
          previewEvidenceArtifact(payload.object.path, artifact),
        );
        actionCell.append(preview);
        firstPreview ||= artifact;
      }
    } else {
      const unavailable = document.createElement("span");
      unavailable.className = "artifact-unavailable";
      unavailable.textContent = artifact.availability.replaceAll("_", " ");
      actionCell.append(unavailable);
    }

    row.append(identityCell, declarationCell, actionCell);
    body.append(row);
  }
  if (firstPreview) {
    previewEvidenceArtifact(payload.object.path, firstPreview);
  }
}

async function selectEvidence(path) {
  state.selectedEvidencePath = path;
  state.selectedEvidence = null;
  renderEvidenceList();
  $("selected-evidence-title").textContent = "Loading…";
  $("selected-evidence-path").textContent = path;
  $("evidence-verify").disabled = true;
  try {
    const payload = await request(`/api/evidence?path=${encodeURIComponent(path)}`);
    if (state.selectedEvidencePath !== path) return;
    state.selectedEvidence = payload;
    const object = payload.object;
    $("selected-evidence-title").textContent = object.id;
    $("selected-evidence-status").textContent = object.status;
    $("selected-evidence-status").className = `status-pill ${statusClass(object.status)}`;
    $("selected-evidence-path").textContent =
      `${object.path} · ${object.object_type} · manifest declarations only`;
    const summary = $("evidence-summary");
    summary.replaceChildren();
    for (const [label, value] of [
      ["Root", object.root_kind],
      ["Registered", object.artifact_count],
      ["Available", object.available_artifact_count],
      ["Declared bytes", formatBytes(object.declared_artifact_bytes)],
      ["Verification", payload.verification.status],
    ]) {
      const item = document.createElement("div");
      const name = document.createElement("span");
      name.textContent = label;
      const content = document.createElement("strong");
      content.textContent = String(value);
      item.append(name, content);
      summary.append(item);
    }
    $("evidence-verify").disabled = false;
    renderEvidenceArtifacts(payload);
  } catch (error) {
    if (state.selectedEvidencePath !== path) return;
    $("selected-evidence-title").textContent = "Could not inspect object";
    $("selected-evidence-status").textContent = "error";
    $("selected-evidence-status").className = "status-pill bad";
    $("selected-evidence-path").textContent = error.message;
    showToast(error.message, true);
  }
}

function selectEvidenceForVerification() {
  const path = state.selectedEvidence?.object?.path;
  if (!path) return;
  const verifier = $("verify-path");
  if (![...verifier.options].some((option) => option.value === path)) {
    showToast("The selected object is no longer in the current catalog.", true);
    return;
  }
  verifier.value = path;
  activateTab("workflows");
  $("verify-form").scrollIntoView({ behavior: "smooth", block: "center" });
  verifier.focus();
  showToast("Selected in the existing verification workflow.");
}

function renderJobs() {
  const body = $("jobs-body");
  body.replaceChildren();
  if (!state.jobs.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "empty-state";
    cell.textContent = "No operator sessions yet.";
    row.append(cell);
    body.append(row);
  }
  for (const job of state.jobs) {
    const row = document.createElement("tr");
    row.className = `session-row ${state.selectedJobId === job.job_id ? "selected" : ""}`;
    row.addEventListener("click", () => selectJob(job.job_id));

    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = `status-pill ${statusClass(job.status)}`;
    status.textContent = job.status;
    statusCell.append(status);

    const titleCell = document.createElement("td");
    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = job.title;
    const id = document.createElement("span");
    id.className = "session-id";
    id.textContent = job.job_id;
    titleCell.append(title, id);

    const timeCell = document.createElement("td");
    timeCell.textContent = readableTime(job.started_at || job.created_at);

    const actionCell = document.createElement("td");
    if (isActive(job.status)) {
      const stop = document.createElement("button");
      stop.type = "button";
      stop.className = "row-action stop";
      stop.textContent = "Stop";
      stop.addEventListener("click", async (event) => {
        event.stopPropagation();
        try {
          await request(`/api/jobs/${encodeURIComponent(job.job_id)}/stop`, {
            method: "POST",
            body: "{}",
          });
          showToast("Stop requested.");
          await refreshJobs();
        } catch (error) {
          showToast(error.message, true);
        }
      });
      actionCell.append(stop);
    } else {
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "row-action";
      inspect.textContent = "Inspect";
      actionCell.append(inspect);
    }
    row.append(statusCell, titleCell, timeCell, actionCell);
    body.append(row);
  }
  const activeCount = state.jobs.filter((job) => isActive(job.status)).length;
  $("jobs-status").textContent = `${activeCount} active job${activeCount === 1 ? "" : "s"}`;
  $("jobs-status").className = `status-pill ${activeCount ? "running" : "neutral"}`;
  $("session-count").textContent = state.jobs.length;
  if (state.selectedJobId) {
    renderInspector();
  }
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  renderJobs();
  await renderInspector();
}

function artifactLink(container, label, objectPath, artifact) {
  if (!artifact.downloadable) return;
  const link = document.createElement("a");
  link.href = artifactUrl(objectPath, artifact.path);
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = label;
  container.append(link);
}

function appendRegisteredArtifactLinks(container, payload, prefix = "") {
  let preview = null;
  for (const artifact of payload.artifacts || []) {
    if (!artifact.downloadable) continue;
    const label = `${prefix}${artifact.role}`;
    artifactLink(container, label, payload.object.path, artifact);
    if (!preview && artifact.preview_kind === "image") {
      preview = {
        objectPath: payload.object.path,
        artifact,
      };
    }
  }
  return preview;
}

async function renderInspector() {
  const job = state.jobs.find((candidate) => candidate.job_id === state.selectedJobId);
  if (!job) return;
  const selectedJobId = job.job_id;
  $("selected-job-title").textContent = job.title;
  $("selected-job-status").textContent = job.status;
  $("selected-job-status").className = `status-pill ${statusClass(job.status)}`;
  $("selected-job-note").textContent = job.error || job.note || "Operator session";
  const links = $("artifact-links");
  links.replaceChildren();
  const image = $("session-preview");
  image.removeAttribute("src");
  image.style.display = "none";
  let preview = null;
  try {
    const session = await request(
      `/api/evidence?path=${encodeURIComponent(`operator_sessions/${job.job_id}`)}`,
    );
    if (state.selectedJobId !== selectedJobId) return;
    preview = appendRegisteredArtifactLinks(links, session);
  } catch (error) {
    if (isActive(job.status)) {
      $("selected-job-note").textContent =
        `${job.note || "Operator session"} · evidence finalization pending`;
    }
  }
  if (job.expected_output_exists && job.expected_output) {
    const workspace = `${state.catalog.workspace.replace(/\/+$/, "")}/`;
    if (job.expected_output.startsWith(workspace)) {
      const relative = job.expected_output.slice(workspace.length);
      try {
        const child = await request(
          `/api/evidence?path=${encodeURIComponent(relative)}`,
        );
        if (state.selectedJobId !== selectedJobId) return;
        preview = appendRegisteredArtifactLinks(links, child, "child · ") || preview;
      } catch (error) {
        if (!isActive(job.status)) {
          $("selected-job-note").textContent =
            `${job.error || job.note || "Operator session"} · child evidence unavailable`;
        }
      }
    }
  }
  if (preview) {
    image.src = `${artifactUrl(preview.objectPath, preview.artifact.path)}&t=${Date.now()}`;
    image.onload = () => {
      image.style.display = "block";
    };
    image.onerror = () => {
      image.style.display = "none";
    };
  }
  try {
    const payload = await request(
      `/api/jobs/${encodeURIComponent(job.job_id)}/log?stream=${state.selectedStream}`,
    );
    $("job-log").textContent = payload.text || "(log is empty)";
    $("job-log").scrollTop = $("job-log").scrollHeight;
  } catch (error) {
    $("job-log").textContent = error.message;
  }
}

async function refreshJobs() {
  try {
    const payload = await request("/api/jobs");
    state.jobs = payload.jobs;
    renderJobs();
  } catch (error) {
    showToast(error.message, true);
  }
}

function bindChrome() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
  }
  for (const tab of document.querySelectorAll(".log-tab")) {
    tab.addEventListener("click", async () => {
      state.selectedStream = tab.dataset.stream;
      for (const candidate of document.querySelectorAll(".log-tab")) {
        candidate.classList.toggle("active", candidate === tab);
      }
      await renderInspector();
    });
  }
  $("refresh-jobs").addEventListener("click", refreshJobs);
  $("evidence-search").addEventListener("input", renderEvidenceList);
  $("evidence-root-filter").addEventListener("change", renderEvidenceList);
  $("evidence-status-filter").addEventListener("change", renderEvidenceList);
  $("evidence-verify").addEventListener("click", selectEvidenceForVerification);
  updateRange("live-confidence", "live-confidence-value", 2);
  updateRange("vehicle-count", "vehicle-count-value");
  updateRange("walker-count", "walker-count-value");
  updateRange("crossing", "crossing-value", 2);
}

async function initialize() {
  bindChrome();
  bindDriveConsole();
  bindLiveForm();
  bindSituationForms();
  bindWorkflowForms();
  try {
    await refreshBootstrap();
  } catch (error) {
    showToast(`Could not initialize operator panel: ${error.message}`, true);
  }
  try {
    await refreshDriveCatalog();
    await refreshDriveState();
  } catch (error) {
    showToast(`Could not initialize Drive Console: ${error.message}`, true);
  }
  window.setInterval(refreshJobs, 1800);
  window.setInterval(refreshDriveState, 350);
}

window.addEventListener("DOMContentLoaded", initialize);
