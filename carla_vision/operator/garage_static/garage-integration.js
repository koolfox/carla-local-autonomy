"use strict";

(() => {
  function savedRunPath() {
    const session = state.drive?.session || {};
    if (session.status !== "success") return "";
    return typeof session.output_path === "string" ? session.output_path : "";
  }

  async function refreshSavedRun(path) {
    await refreshBootstrap();
    const exists = (state.catalog?.research_objects || []).some((row) => row.path === path);
    if (!exists) {
      throw new Error(`Saved run ${path} is not present in the research catalog.`);
    }
  }

  async function inspectSavedRun(button) {
    const path = savedRunPath();
    if (!path) {
      showToast("Finish and save a drive before opening it in Research.", true);
      return;
    }
    button.disabled = true;
    try {
      await refreshSavedRun(path);
      activateTab("tools");
      activateResearchTool("evidence");
      await selectEvidence(path);
      document.getElementById("panel-evidence")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      showToast("Opened this drive in Saved results.");
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function verifySavedRun(button) {
    const path = savedRunPath();
    if (!path) {
      showToast("Finish and save a drive before verifying it.", true);
      return;
    }
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

  function installGarageResearchActions() {
    const banner = document.getElementById("drive-result-banner");
    if (!banner || document.getElementById("drive-research-actions")) return;

    const panel = document.createElement("div");
    panel.id = "drive-research-actions";
    panel.className = "drive-research-actions";

    const copy = document.createElement("div");
    copy.className = "drive-research-copy";
    const title = document.createElement("strong");
    title.textContent = "Continue with this run";
    const note = document.createElement("span");
    note.textContent = "Inspect the exact saved artifacts or run the existing verifier.";
    copy.append(title, note);

    const actions = document.createElement("div");
    actions.className = "drive-research-buttons";

    const inspect = document.createElement("button");
    inspect.id = "drive-inspect-run";
    inspect.type = "button";
    inspect.className = "secondary";
    inspect.textContent = "Inspect saved run";
    inspect.addEventListener("click", () => void inspectSavedRun(inspect));

    const verify = document.createElement("button");
    verify.id = "drive-verify-run";
    verify.type = "button";
    verify.className = "secondary";
    verify.textContent = "Verify saved run";
    verify.addEventListener("click", () => void verifySavedRun(verify));

    actions.append(inspect, verify);
    panel.append(copy, actions);
    banner.append(panel);
  }

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", installGarageResearchActions, { once: true });
  } else {
    installGarageResearchActions();
  }
})();
