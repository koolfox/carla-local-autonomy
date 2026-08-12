"use strict";

(() => {
  const workflows = {
    teacher_capture: {
      label: "Record BehaviorAgent teacher",
      source: "scenario",
      amountLabel: "Max episodes",
      amount: 3,
      ack: "This reloads scenario maps and takes exclusive ownership of CARLA ticks.",
    },
    imitation_train: {
      label: "Train imitation driver",
      source: "dataset",
      amountLabel: "Epochs",
      amount: 15,
      device: true,
      dry: true,
    },
    voxel_capture: {
      label: "Capture voxel teacher",
      source: "none",
      amountLabel: "Frames",
      amount: 120,
      dry: true,
      live: true,
    },
    voxel_rgb_capture: {
      label: "Capture RGB voxel predictions",
      backendKind: "voxel_capture",
      source: "none",
      checkpoint: true,
      amountLabel: "Frames",
      amount: 120,
      device: true,
      dry: true,
      live: true,
    },
    voxel_flow_capture: {
      label: "Capture voxel flow teacher",
      source: "none",
      amountLabel: "Frames",
      amount: 120,
      dry: true,
      live: true,
    },
    voxel_train: {
      label: "Train voxel occupancy",
      source: "dataset",
      amountLabel: "Epochs",
      amount: 10,
      device: true,
      dry: true,
    },
    voxel_flow_train: {
      label: "Train voxel + flow",
      source: "dataset",
      amountLabel: "Epochs",
      amount: 10,
      device: true,
      dry: true,
    },
    voxel_shadow: {
      label: "Run voxel shadow planner",
      source: "none",
      checkpoint: true,
      amountLabel: "Frames",
      amount: 300,
      device: true,
      dry: true,
      live: true,
    },
    voxel_benchmark: {
      label: "Benchmark recorded voxel run",
      source: "run",
      amountLabel: "",
    },
    closed_loop_evaluate: {
      label: "Observe closed-loop behavior",
      source: "none",
      amountLabel: "Seconds",
      amount: 120,
      dry: true,
      live: true,
    },
  };

  const $garage = (id) => document.getElementById(id);

  function valuesForSource(kind) {
    if (kind === "scenario") return state.catalog?.scenario_plans || [];
    if (kind === "dataset") return state.catalog?.datasets || [];
    if (kind === "run") {
      return (state.catalog?.research_objects || [])
        .filter((row) => row.root_kind === "runs")
        .map((row) => row.path);
    }
    return [];
  }

  function setSelectValues(select, values, emptyLabel) {
    const previous = select.value;
    select.replaceChildren();
    if (!values.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = emptyLabel;
      select.append(option);
      return;
    }
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.append(option);
    }
    if ([...select.options].some((option) => option.value === previous)) select.value = previous;
  }

  function defaultResearchId(kind) {
    const prefix = {
      teacher_capture: "teacher-ds",
      imitation_train: "imitation",
      voxel_capture: "voxel-capture",
      voxel_rgb_capture: "voxel-rgb",
      voxel_flow_capture: "voxel-flow-capture",
      voxel_train: "voxel",
      voxel_flow_train: "voxel-flow",
      voxel_shadow: "voxel-shadow",
      voxel_benchmark: "voxel-benchmark",
      closed_loop_evaluate: "closed-loop",
    }[kind] || "research";
    return generatedId(prefix);
  }

  function liveHeroAvailable() {
    return driveIsRunning() && Boolean(state.drive?.session?.vehicle_id);
  }

  function syncResearchLauncher() {
    const select = $garage("garage-research-kind");
    if (!select) return;
    const kind = select.value;
    const workflow = workflows[kind];
    if (!workflow) return;

    const sourceWrap = $garage("garage-research-source-wrap");
    const source = $garage("garage-research-source");
    sourceWrap.hidden = workflow.source === "none";
    if (workflow.source !== "none") {
      setSelectValues(source, valuesForSource(workflow.source), `No ${workflow.source} available`);
      $garage("garage-research-source-label").textContent = {
        scenario: "Scenario plan",
        dataset: "Dataset",
        run: "Recorded run",
      }[workflow.source];
    }

    const checkpointWrap = $garage("garage-research-checkpoint-wrap");
    checkpointWrap.hidden = !workflow.checkpoint;
    if (workflow.checkpoint) {
      setSelectValues(
        $garage("garage-research-checkpoint"),
        state.drive?.catalog?.policy_checkpoints || [],
        "No checkpoint available",
      );
    }

    $garage("garage-research-device-wrap").hidden = !workflow.device;
    $garage("garage-research-amount-wrap").hidden = !workflow.amountLabel;
    if (workflow.amountLabel) {
      $garage("garage-research-amount-label").textContent = workflow.amountLabel;
      if ($garage("garage-research-amount").dataset.kind !== kind) {
        $garage("garage-research-amount").value = String(workflow.amount);
        $garage("garage-research-amount").dataset.kind = kind;
      }
    }
    $garage("garage-research-dry-wrap").hidden = !workflow.dry;
    const ackWrap = $garage("garage-research-ack-wrap");
    ackWrap.hidden = !workflow.ack;
    $garage("garage-research-ack-note").textContent = workflow.ack || "";

    const live = $garage("garage-research-live-state");
    if (workflow.live) {
      live.hidden = false;
      live.textContent = liveHeroAvailable()
        ? `Live ego ${state.drive.session.vehicle_id} available`
        : "Start a Garage drive first for a real live run. Dry-run remains available.";
      live.className = `garage-research-live ${liveHeroAvailable() ? "ready" : "waiting"}`;
    } else {
      live.hidden = true;
    }

    const runId = $garage("garage-research-run-id");
    if (runId.dataset.kind !== kind) {
      runId.value = defaultResearchId(kind);
      runId.dataset.kind = kind;
    }
  }

  function researchPayload(kind) {
    const workflow = workflows[kind];
    const backendKind = workflow.backendKind || kind;
    const source = $garage("garage-research-source").value;
    const runId = $garage("garage-research-run-id").value.trim();
    const amount = Number($garage("garage-research-amount").value || workflow.amount || 1);
    const device = $garage("garage-research-device").value;
    const dryRun = workflow.dry && $garage("garage-research-dry").checked;
    const checkpoint = $garage("garage-research-checkpoint").value;

    if (kind === "teacher_capture") {
      return {
        kind: backendKind,
        parameters: {
          scenario_plan: source,
          dataset_id: runId,
          behavior: $garage("garage-research-behavior").value,
          max_episodes: amount,
          acknowledge: $garage("garage-research-ack").checked,
        },
      };
    }
    if (["imitation_train", "voxel_train", "voxel_flow_train"].includes(kind)) {
      return {
        kind: backendKind,
        parameters: {
          dataset: source,
          run_id: runId,
          device,
          epochs: amount,
          dry_run: dryRun,
        },
      };
    }
    if (["voxel_capture", "voxel_rgb_capture", "voxel_flow_capture"].includes(kind)) {
      const parameters = { run_id: runId, frames: amount, dry_run: dryRun };
      if (kind === "voxel_capture") parameters.mode = "teacher";
      if (kind === "voxel_rgb_capture") {
        parameters.mode = "rgb-only";
        parameters.checkpoint = checkpoint;
        parameters.device = device;
      }
      return { kind: backendKind, parameters };
    }
    if (kind === "voxel_shadow") {
      return {
        kind: backendKind,
        parameters: {
          run_id: runId,
          checkpoint,
          device,
          frames: amount,
          dry_run: dryRun,
        },
      };
    }
    if (kind === "voxel_benchmark") {
      return { kind: backendKind, parameters: { run: source, run_id: runId } };
    }
    if (kind === "closed_loop_evaluate") {
      return {
        kind: backendKind,
        parameters: {
          run_id: runId,
          driver_label:
            state.drive?.session?.garage_mode ||
            state.drive?.session?.control_mode ||
            "unknown",
          duration: amount,
          dry_run: dryRun,
        },
      };
    }
    throw new Error(`Unsupported research workflow ${kind}`);
  }

  async function launchResearch() {
    const button = $garage("garage-research-launch");
    const kind = $garage("garage-research-kind").value;
    const workflow = workflows[kind];
    const dryRun = workflow.dry && $garage("garage-research-dry").checked;
    if (workflow.live && !dryRun && !liveHeroAvailable()) {
      showToast("This live workflow needs a running Garage ego vehicle.", true);
      return;
    }
    button.disabled = true;
    try {
      const built = researchPayload(kind);
      const job = await request("/api/garage/jobs", {
        method: "POST",
        body: JSON.stringify({
          schema_version: "1.0",
          kind: built.kind,
          parameters: built.parameters,
        }),
      });
      state.selectedJobId = job.job_id;
      showToast(`${job.title} queued.`);
      await refreshJobs();
      activateTab("tools");
      activateResearchTool("sessions");
      $garage("garage-research-run-id").dataset.kind = "";
      syncResearchLauncher();
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function installResearchLauncher() {
    const shelf = $garage("garage-research-shelf");
    if (!shelf || $garage("garage-research-launcher")) return;
    const launcher = document.createElement("details");
    launcher.id = "garage-research-launcher";
    launcher.className = "garage-research-launcher";
    const summary = document.createElement("summary");
    summary.textContent = "Run research workflow";
    const body = document.createElement("div");
    body.className = "garage-research-launcher-body";
    body.innerHTML = `
      <label class="span-2">
        Workflow
        <select id="garage-research-kind"></select>
      </label>
      <label class="span-2" id="garage-research-source-wrap">
        <span id="garage-research-source-label">Source</span>
        <select id="garage-research-source"></select>
      </label>
      <label class="span-2" id="garage-research-checkpoint-wrap" hidden>
        Checkpoint
        <select id="garage-research-checkpoint"></select>
      </label>
      <label>
        Run / output name
        <input id="garage-research-run-id" autocomplete="off">
      </label>
      <label id="garage-research-amount-wrap">
        <span id="garage-research-amount-label">Amount</span>
        <input id="garage-research-amount" type="number" min="1" value="10">
      </label>
      <label id="garage-research-device-wrap" hidden>
        Device
        <select id="garage-research-device">
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
          <option value="mps">MPS</option>
        </select>
      </label>
      <label id="garage-research-behavior-wrap">
        Teacher style
        <select id="garage-research-behavior">
          <option value="cautious">Cautious</option>
          <option value="normal" selected>Normal</option>
          <option value="aggressive">Aggressive</option>
        </select>
      </label>
      <label id="garage-research-dry-wrap" class="garage-research-check">
        <input id="garage-research-dry" type="checkbox">
        <span>Dry-run only</span>
      </label>
      <label id="garage-research-ack-wrap" class="garage-research-check" hidden>
        <input id="garage-research-ack" type="checkbox">
        <span id="garage-research-ack-note"></span>
      </label>
      <p id="garage-research-live-state" class="garage-research-live" hidden></p>
      <button id="garage-research-launch" class="primary span-2" type="button">Run workflow</button>
    `;
    launcher.append(summary, body);
    shelf.append(launcher);

    const kindSelect = $garage("garage-research-kind");
    for (const [kind, workflow] of Object.entries(workflows)) {
      const option = document.createElement("option");
      option.value = kind;
      option.textContent = workflow.label;
      kindSelect.append(option);
    }
    kindSelect.addEventListener("change", syncResearchLauncher);
    $garage("garage-research-launch").addEventListener("click", () => void launchResearch());
    window.setInterval(syncResearchLauncher, 500);
    syncResearchLauncher();
  }

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", installResearchLauncher, { once: true });
  } else {
    installResearchLauncher();
  }
})();
