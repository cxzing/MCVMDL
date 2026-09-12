const $ = (id) => document.getElementById(id);

const state = {
  plans: [],
  plan: null,
  profile: null,
  selectedTissueLabel: null,
  job: null,
  axis: "x",
  pollTimer: null,
};

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ? `${data.error}: ${data.detail}` : data.error || `HTTP ${response.status}`);
  return data;
}

async function postJson(url, payload) {
  return jsonRequest(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
}

function setJobStatus(title, detail, progress = null) {
  $("jobStatus").textContent = title;
  $("jobDetail").textContent = detail;
  if (progress !== null) $("jobProgress").value = Math.max(0, Math.min(1, progress));
}

function selectedPlanId() {
  return $("planSelect").value;
}

function requiredTissueLabels() {
  return (state.plan?.tissueLabels || []).map(Number).filter((label) => label > 0);
}

function maximumTissueLabel() {
  return Math.max(1, Number(state.plan?.config?.numChannels || 17) - 1);
}

function selectedTissue() {
  return state.profile?.tissues?.find((tissue) => Number(tissue.label) === Number(state.selectedTissueLabel)) || null;
}

function updateTissueSummary() {
  const rows = state.profile?.tissues || [];
  const required = requiredTissueLabels();
  const defined = new Set(rows.map((row) => Number(row.label)));
  const covered = required.filter((label) => defined.has(label)).length;
  $("tissueSummaryText").textContent = `${rows.length} definitions · ${covered}/${required.length} required labels covered`;
  $("addTissueBtn").disabled = rows.length >= maximumTissueLabel();
}

function fillTissueEditor() {
  const tissue = selectedTissue();
  const fields = ["tissueLabel", "tissueName", "tissueN", "tissueMua", "tissueMus", "tissueG"];
  fields.forEach((id) => { $(id).disabled = !tissue; });
  $("deleteTissueBtn").disabled = !tissue || (state.profile?.tissues?.length || 0) <= 1;
  if (!tissue) {
    fields.forEach((id) => { $(id).value = ""; });
    $("tissuePresenceText").textContent = "Add a tissue definition to continue.";
    return;
  }
  $("tissueLabel").value = tissue.label;
  $("tissueName").value = tissue.name;
  $("tissueN").value = tissue.n;
  $("tissueMua").value = tissue.mua;
  $("tissueMus").value = tissue.mus;
  $("tissueG").value = tissue.g;
  const present = requiredTissueLabels().includes(Number(tissue.label));
  $("tissuePresenceText").textContent = present
    ? `Label ${tissue.label} is present in the selected tissue volume and is required.`
    : `Label ${tissue.label} is not present in the selected volume and is optional.`;
}

function renderTissueSelector(preferredLabel = null) {
  const rows = state.profile?.tissues || [];
  rows.sort((left, right) => Number(left.label) - Number(right.label));
  const required = new Set(requiredTissueLabels());
  const select = $("tissueSelect");
  select.innerHTML = "";
  rows.forEach((tissue) => {
    const option = document.createElement("option");
    option.value = String(tissue.label);
    option.textContent = `${tissue.label} · ${tissue.name}${required.has(Number(tissue.label)) ? " · present" : ""}`;
    select.appendChild(option);
  });
  const preferred = Number(preferredLabel);
  if (rows.some((row) => Number(row.label) === preferred)) state.selectedTissueLabel = preferred;
  else state.selectedTissueLabel = rows.length ? Number(rows[0].label) : null;
  if (state.selectedTissueLabel !== null) select.value = String(state.selectedTissueLabel);
  select.disabled = !rows.length;
  updateTissueSummary();
  fillTissueEditor();
}

function profileFromForm() {
  if (!state.profile) throw new Error("Optical profile is not loaded.");
  const tissues = state.profile.tissues.map((tissue) => ({
    label: Number(tissue.label),
    name: String(tissue.name || `Tissue ${tissue.label}`),
    n: Number(tissue.n),
    mua: Number(tissue.mua),
    mus: Number(tissue.mus),
    g: Number(tissue.g),
  }));
  return {
    profileVersion: state.profile.profileVersion || "custom",
    verificationMode: state.profile.verificationMode,
    voxelSizeMm: [Number($("voxelX").value), Number($("voxelY").value), Number($("voxelZ").value)],
    outsideRefractiveIndex: Number($("outsideN").value),
    beam: {
      radiusCm: Number($("beamRadius").value),
      profile: $("beamProfile").value,
      fiberNA: Number($("fiberNA").value),
    },
    noiseThreshold: Number($("noiseThreshold").value),
    tissues,
  };
}

function renderProfile(profile) {
  state.profile = JSON.parse(JSON.stringify(profile));
  [$("voxelX").value, $("voxelY").value, $("voxelZ").value] = state.profile.voxelSizeMm;
  $("outsideN").value = state.profile.outsideRefractiveIndex;
  $("beamRadius").value = state.profile.beam.radiusCm;
  $("beamProfile").value = state.profile.beam.profile;
  $("fiberNA").value = state.profile.beam.fiberNA;
  $("noiseThreshold").value = state.profile.noiseThreshold;
  renderTissueSelector(state.profile.tissues[0]?.label);
}

function updatePlannedSourceSummary() {
  const best = state.plan?.bestCandidate;
  const root = $("plannedSourceSummary");
  root.innerHTML = "";
  if (!best) return;
  const values = [
    ["Candidate", best.id],
    ["Source (model grid)", best.source.map((value) => Number(value).toFixed(1)).join(", ")],
    ["Direction", best.direction.map((value) => Number(value).toFixed(3)).join(", ")],
    ["ROI", state.plan.config.roi.center.map((value) => Number(value).toFixed(0)).join(", ")],
  ];
  values.forEach(([label, value]) => {
    const item = document.createElement("span");
    const name = document.createElement("small"); name.textContent = label;
    const strong = document.createElement("strong"); strong.textContent = value;
    item.append(name, strong); root.appendChild(item);
  });
}

function initializeManualSource(plan) {
  const best = plan.bestCandidate;
  const aim = best.aimTarget || plan.config.roi.center;
  [$("manualSourceX").value, $("manualSourceY").value, $("manualSourceZ").value] = best.source;
  [$("manualAimX").value, $("manualAimY").value, $("manualAimZ").value] = aim;
  const shape = plan.shape || [128, 128, 128];
  ["manualSourceX", "manualAimX"].forEach((id) => { $(id).min = 0; $(id).max = shape[0] - 1; });
  ["manualSourceY", "manualAimY"].forEach((id) => { $(id).min = 0; $(id).max = shape[1] - 1; });
  ["manualSourceZ", "manualAimZ"].forEach((id) => { $(id).min = 0; $(id).max = shape[2] - 1; });
  updateManualDirection();
}

function manualCoordinates() {
  return {
    source: [Number($("manualSourceX").value), Number($("manualSourceY").value), Number($("manualSourceZ").value)],
    aimTarget: [Number($("manualAimX").value), Number($("manualAimY").value), Number($("manualAimZ").value)],
  };
}

function updateManualDirection() {
  const { source, aimTarget } = manualCoordinates();
  const direction = aimTarget.map((value, index) => value - source[index]);
  const norm = Math.hypot(...direction);
  $("manualDirectionText").textContent = Number.isFinite(norm) && norm > 1e-8
    ? direction.map((value) => (value / norm).toFixed(3)).join(", ")
    : "Source and aim must differ";
}

function sourceSelectionFromForm() {
  if ($("sourceMode").value === "planned") return { mode: "planned" };
  return { mode: "manual", ...manualCoordinates() };
}

function renderSourceMode(selection = null) {
  const mode = selection?.mode === "manual" ? "manual" : "planned";
  $("sourceMode").value = mode;
  if (selection?.source) {
    [$("manualSourceX").value, $("manualSourceY").value, $("manualSourceZ").value] = selection.source;
  }
  if (selection?.aimTarget) {
    [$("manualAimX").value, $("manualAimY").value, $("manualAimZ").value] = selection.aimTarget;
  }
  $("manualSourceFields").hidden = mode !== "manual";
  $("plannedSourceSummary").hidden = mode === "manual";
  updateManualDirection();
}

async function loadPlan(planId, resetProfile = true) {
  if (!planId) return;
  const plan = await jsonRequest(`/api/plans/${planId}`);
  if (plan.status !== "completed") throw new Error("The selected plan has not completed.");
  state.plan = plan;
  localStorage.setItem("mcvm.currentPlanId", planId);
  updatePlannedSourceSummary();
  initializeManualSource(plan);
  renderSourceMode({ mode: "planned" });
  if (resetProfile) renderProfile(await jsonRequest(`/api/mcvm/profile?planId=${encodeURIComponent(planId)}`));
  const compatible = state.profile?.compatibleTissue !== false;
  setJobStatus(
    compatible ? "Plan ready" : "Tissue volume incompatible",
    compatible
      ? `Selected ${plan.bestCandidate.id}. Source coordinates map directly to the ${state.profile.tissueShape?.join(" × ") || "selected"} MCVM grid.`
      : "This plan does not use a compatible three-dimensional tissue volume.",
    0,
  );
  $("runMcvmBtn").disabled = !compatible;
}

async function loadPlans() {
  const data = await jsonRequest("/api/plans");
  state.plans = data.plans.filter((plan) => plan.status === "completed");
  const select = $("planSelect");
  select.innerHTML = "";
  if (!state.plans.length) {
    const option = document.createElement("option"); option.value = ""; option.textContent = "No completed plans";
    select.appendChild(option); $("runMcvmBtn").disabled = true; return;
  }
  state.plans.forEach((plan) => {
    const option = document.createElement("option"); option.value = plan.planId;
    option.textContent = `${plan.planId} · ${plan.bestCandidate?.candidate?.id || plan.bestCandidate?.id || "completed"}`;
    select.appendChild(option);
  });
  const current = localStorage.getItem("mcvm.currentPlanId");
  if (current && state.plans.some((plan) => plan.planId === current)) select.value = current;
  await loadPlan(select.value);
}

async function restoreLatestJobForPlan(planId) {
  const data = await jsonRequest("/api/mcvm/jobs");
  const compatible = data.jobs.find((job) =>
    job.planId === planId
    && job.status === "completed"
    && job.profileVersion === state.profile?.profileVersion
    && Number(job.mciVersion) === 2
  );
  const match = compatible || data.jobs.find((job) => job.planId === planId && job.status === "completed");
  if (!match) return false;
  const job = await jsonRequest(`/api/mcvm/jobs/${match.jobId}`);
  state.job = job;
  localStorage.setItem("mcvm.currentJobId", job.jobId);
  const compatibleProfile = Boolean(
    job.profile?.profileVersion
    && state.profile?.profileVersion
    && job.profile.profileVersion === state.profile.profileVersion
  );
  if (compatibleProfile) renderProfile(job.profile);
  renderSourceMode(job.sourceSelection || { mode: "planned" });
  renderResults(job);
  setJobStatus(
    "Historical verification loaded",
    compatibleProfile
      ? `Loaded completed MCVM job ${job.jobId}.`
      : `Loaded result ${job.jobId}; its legacy optical profile is shown only in the saved job. New runs use the current model-compatible defaults.`,
    1,
  );
  return true;
}

function selectedPhotons() {
  return Number($("photonPreset").value);
}

async function startVerification() {
  if (!state.plan) throw new Error("Select a completed plan first.");
  const photons = selectedPhotons();
  if (photons >= 10000000 && !window.confirm("10⁷-photon MCVM verification takes approximately one hour and may temporarily use several gigabytes of disk space. Start now?")) return;
  $("runMcvmBtn").disabled = true;
  try {
    const job = await postJson("/api/mcvm/jobs", {
      planId: state.plan.planId,
      photons,
      retainRaw: $("retainRaw").checked,
      profile: profileFromForm(),
      sourceSelection: sourceSelectionFromForm(),
    });
    state.job = job;
    localStorage.setItem("mcvm.currentJobId", job.jobId);
    $("cancelMcvmBtn").disabled = false;
    $("verificationResults").hidden = true;
    const sourceText = job.sourceSelection?.mode === "manual" ? "manual source" : state.plan.bestCandidate.id;
    setJobStatus("MCVM queued", `Mapping ${sourceText} directly to the selected tissue grid for job ${job.jobId}.`, 0);
    pollJob(job.jobId);
  } catch (error) {
    $("runMcvmBtn").disabled = false;
    setJobStatus("Could not start MCVM", error.message, 0);
  }
}

function pollJob(jobId) {
  window.clearTimeout(state.pollTimer);
  const tick = async () => {
    try {
      const job = await jsonRequest(`/api/mcvm/jobs/${jobId}`);
      state.job = job;
      const fraction = Number(job.progress?.fraction || 0);
      const phase = job.progress?.phase || "simulation";
      const eta = job.progress?.estimatedEnd ? ` · MCVM estimate ${job.progress.estimatedEnd}` : "";
      if (job.status === "running" && ["compatibilityCheck", "compatibilitySimulation"].includes(phase)) {
        setJobStatus(
          "Checking MCVM v2 compatibility",
          "A cached 20-photon check confirms the v2 input and aligned output before the requested run.",
          fraction,
        );
      } else if (job.status === "running" && ["startingSimulation", "inputValidated"].includes(phase)) {
        setJobStatus(
          phase === "inputValidated" ? "MCVM v2 input accepted" : "Starting same-grid MCVM",
          "The selected tissue and optical parameters passed the input check.",
          0,
        );
      } else if (job.status === "running" && phase === "writingOutputs") {
        setJobStatus(
          "Writing MCVM field",
          `Photon propagation has finished; MCVM is writing the ${job.shape?.join(" × ") || "aligned"} absorption field.`,
          1,
        );
      } else if (job.status === "processing" || phase === "normalization") {
        setJobStatus(
          "Normalizing MCVM output",
          "Restoring x–z–y axes and applying the model log10 normalization on the same grid.",
          1,
        );
      } else {
        setJobStatus(
          job.status === "running" ? "MCVM simulation running" : `MCVM ${job.status}`,
          `${Number(job.photons).toLocaleString()} photons · ${(fraction * 100).toFixed(2)}% reported${eta}`,
          fraction,
        );
      }
      if (job.status === "completed") {
        $("runMcvmBtn").disabled = false;
        $("cancelMcvmBtn").disabled = true;
        setJobStatus("Verification completed", "The same-grid MCVM field is ready for comparison.", 1);
        renderResults(job);
        return;
      }
      if (["failed", "cancelled", "interrupted"].includes(job.status)) {
        $("runMcvmBtn").disabled = false;
        $("cancelMcvmBtn").disabled = true;
        setJobStatus(`MCVM ${job.status}`, job.error?.detail || job.error?.message || "The job did not complete.", fraction);
        return;
      }
      state.pollTimer = window.setTimeout(tick, 1000);
    } catch (error) {
      $("runMcvmBtn").disabled = false;
      setJobStatus("Job status error", error.message);
    }
  };
  tick();
}

async function cancelVerification() {
  if (!state.job?.jobId) return;
  $("cancelMcvmBtn").disabled = true;
  try {
    await postJson(`/api/mcvm/jobs/${state.job.jobId}/cancel`, {});
    setJobStatus("Cancelling MCVM", "Stopping the process and removing incomplete raw outputs…", state.job.progress?.fraction || 0);
  } catch (error) {
    setJobStatus("Cancellation failed", error.message);
  }
}

function formatMetric(value, percent = false) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  if (!percent) return Number(value).toExponential(3);
  const percentage = Number(value) * 100;
  return Math.abs(percentage) >= 10000 ? `${percentage.toExponential(2)}%` : `${percentage.toFixed(2)}%`;
}

function renderResults(job) {
  $("verificationResults").hidden = false;
  const photonValue = String(job.photons || "");
  if ([...$("photonPreset").options].some((option) => option.value === photonValue)) {
    $("photonPreset").value = photonValue;
  }
  const mode = job.sourceSelection?.mode === "manual" ? "manual source" : (job.sourceSelection?.candidateId || "planned source");
  const fidelity = job.verificationMode === "sameGrid" && Number(job.mciVersion) === 2
    ? "same-grid · MCI v2"
    : "archived legacy result";
  $("verificationDuration").textContent = `${mode} · ${fidelity} · ${Number(job.photons).toLocaleString()} photons · ${Number(job.durationSec).toFixed(1)} s`;
  const metrics = job.metrics;
  const physical = metrics.physicalSpatial || metrics.spatial;
  const items = [
    ["Target relative error", formatMetric(metrics.relativeError.targetAbsorption, true)],
    ["Off-target relative error", formatMetric(metrics.relativeError.offTargetExposure, true)],
    ["Hotspot relative error", formatMetric(metrics.relativeError.hotspotRisk, true)],
    ["Log Pearson", metrics.spatial.pearson == null ? "-" : Number(metrics.spatial.pearson).toFixed(4)],
    ["Log NRMSE", formatMetric(metrics.spatial.nrmse)],
    ["Physical L2", formatMetric(physical.relativeL2)],
    ["Physical MAE", formatMetric(physical.mae)],
    ["MCVM target", formatMetric(metrics.mcvm.targetAbsorption)],
  ];
  const root = $("verificationMetrics"); root.innerHTML = "";
  items.forEach(([label, value]) => {
    const card = document.createElement("div"); card.className = "metric-card";
    const span = document.createElement("span"); span.textContent = label;
    const strong = document.createElement("strong"); strong.textContent = value;
    card.append(span, strong); root.appendChild(card);
  });
  const center = state.plan?.config?.roi?.center || [64, 64, 64];
  setAxis(state.axis, Math.round(center[{ x: 0, y: 1, z: 2 }[state.axis]]));
  loadResultReference(job.jobId).catch((error) => setJobStatus("Reference-point error", error.message, 1));
}

async function loadResultReference(jobId) {
  const meta = await jsonRequest(`/api/mcvm/jobs/${jobId}/comparison-meta`);
  if (state.job?.jobId !== jobId) return;
  const labels = { mcvmPeak: "MCVM peak", modelPeak: "model peak", roiCenter: "ROI center" };
  $("resultReferenceText").textContent = `Reference: ${labels[meta.referenceMode] || meta.referenceMode} [${meta.referencePoint.join(", ")}] · shared max ${Number(meta.commonMax).toExponential(2)}`;
  const dimension = { x: 0, y: 1, z: 2 }[state.axis];
  setAxis(state.axis, meta.referencePoint[dimension]);
}

async function loadVerificationSlice(kind, canvas) {
  if (!state.job?.jobId || state.job.status !== "completed") return;
  const index = Number($("verificationSlice").value);
  const response = await fetch(`/api/mcvm/jobs/${state.job.jobId}/slice?kind=${kind}&axis=${state.axis}&index=${index}&overlays=1`);
  if (!response.ok) { const error = await response.json(); throw new Error(error.detail ? `${error.error}: ${error.detail}` : error.error); }
  const width = Number(response.headers.get("X-Width"));
  const height = Number(response.headers.get("X-Height"));
  const bytes = new Uint8ClampedArray(await response.arrayBuffer());
  canvas.width = width; canvas.height = height;
  canvas.getContext("2d", { alpha: false }).putImageData(new ImageData(bytes, width, height), 0, 0);
}

async function loadAllVerificationSlices() {
  $("verificationSliceLabel").textContent = `${state.axis.toUpperCase()} ${$("verificationSlice").value}`;
  await Promise.all([
    loadVerificationSlice("model", $("modelCanvas")),
    loadVerificationSlice("mcvm", $("mcvmCanvas")),
    loadVerificationSlice("difference", $("differenceCanvas")),
  ]);
}

function setAxis(axis, value = null) {
  state.axis = axis;
  document.querySelectorAll("[data-axis]").forEach((button) => button.classList.toggle("active", button.dataset.axis === axis));
  const dimension = { x: 0, y: 1, z: 2 }[axis];
  const shape = state.plan?.shape || [128, 128, 128];
  $("verificationSlice").max = shape[dimension] - 1;
  if (value !== null) $("verificationSlice").value = Math.max(0, Math.min(shape[dimension] - 1, value));
  loadAllVerificationSlices().catch((error) => setJobStatus("Slice error", error.message, 1));
}

function exportProfile() {
  const blob = new Blob([JSON.stringify(profileFromForm(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a"); link.href = url; link.download = `mcvm-optical-profile-${state.plan?.planId || "custom"}.json`;
  link.click(); URL.revokeObjectURL(url);
}

async function importProfile(file) {
  renderProfile(JSON.parse(await file.text()));
  setJobStatus("Profile imported", `${file.name} loaded. Review values before starting MCVM.`, 0);
}

function addTissue() {
  const rows = state.profile?.tissues || [];
  const used = new Set(rows.map((row) => Number(row.label)));
  let label = 1;
  while (used.has(label) && label <= maximumTissueLabel()) label += 1;
  if (label > maximumTissueLabel()) {
    setJobStatus("Cannot add tissue", `All ${maximumTissueLabel()} model tissue labels are already defined.`, 0);
    return;
  }
  rows.push({ label, name: `Tissue ${label}`, n: 1.37, mua: 0.1, mus: 100, g: 0.9 });
  renderTissueSelector(label);
  setJobStatus("Tissue added", `Added editable label ${label}.`, 0);
}

function deleteTissue() {
  const tissue = selectedTissue();
  if (!tissue || state.profile.tissues.length <= 1) return;
  state.profile.tissues = state.profile.tissues.filter((row) => Number(row.label) !== Number(tissue.label));
  const wasRequired = requiredTissueLabels().includes(Number(tissue.label));
  renderTissueSelector();
  setJobStatus(
    "Tissue removed",
    wasRequired ? `Label ${tissue.label} is used by this volume and must be re-added before verification.` : `Removed optional label ${tissue.label}.`,
    0,
  );
}

function updateTissueField(field, value) {
  const tissue = selectedTissue();
  if (!tissue) return;
  tissue[field] = field === "name" ? value : Number(value);
  if (field === "name") renderTissueSelector(tissue.label);
}

function updateTissueLabel() {
  const tissue = selectedTissue();
  if (!tissue) return;
  const previous = Number(tissue.label);
  const next = Number($("tissueLabel").value);
  const duplicate = state.profile.tissues.some((row) => row !== tissue && Number(row.label) === next);
  if (!Number.isInteger(next) || next < 1 || next > maximumTissueLabel() || duplicate) {
    $("tissueLabel").value = previous;
    setJobStatus("Invalid tissue label", `Use a unique integer from 1 to ${maximumTissueLabel()}.`, 0);
    return;
  }
  tissue.label = next;
  renderTissueSelector(next);
}

function debounce(fn, delay = 100) {
  let timer; return (...args) => { window.clearTimeout(timer); timer = window.setTimeout(() => fn(...args), delay); };
}

$("planSelect").addEventListener("change", async () => {
  try {
    await loadPlan(selectedPlanId());
    if (!(await restoreLatestJobForPlan(selectedPlanId()))) {
      state.job = null;
      $("verificationResults").hidden = true;
      localStorage.removeItem("mcvm.currentJobId");
    }
  } catch (error) {
    setJobStatus("Plan error", error.message);
  }
});
$("sourceMode").addEventListener("change", () => renderSourceMode({ mode: $("sourceMode").value }));
["manualSourceX", "manualSourceY", "manualSourceZ", "manualAimX", "manualAimY", "manualAimZ"].forEach((id) => $(id).addEventListener("input", updateManualDirection));
$("runMcvmBtn").addEventListener("click", () => startVerification().catch((error) => setJobStatus("Could not start MCVM", error.message)));
$("cancelMcvmBtn").addEventListener("click", cancelVerification);
$("importProfileBtn").addEventListener("click", () => $("profileFileInput").click());
$("exportProfileBtn").addEventListener("click", exportProfile);
$("resetProfileBtn").addEventListener("click", async () => {
  try { renderProfile(await jsonRequest(`/api/mcvm/profile?planId=${encodeURIComponent(selectedPlanId())}`)); setJobStatus("Defaults restored", "Optical defaults were restored.", 0); }
  catch (error) { setJobStatus("Profile error", error.message); }
});
$("profileFileInput").addEventListener("change", (event) => { const file = event.target.files?.[0]; if (file) importProfile(file).catch((error) => setJobStatus("Profile import failed", error.message)); event.target.value = ""; });
$("tissueSelect").addEventListener("change", () => { state.selectedTissueLabel = Number($("tissueSelect").value); fillTissueEditor(); });
$("addTissueBtn").addEventListener("click", addTissue);
$("deleteTissueBtn").addEventListener("click", deleteTissue);
$("tissueLabel").addEventListener("change", updateTissueLabel);
$("tissueName").addEventListener("input", () => updateTissueField("name", $("tissueName").value));
$("tissueN").addEventListener("input", () => updateTissueField("n", $("tissueN").value));
$("tissueMua").addEventListener("input", () => updateTissueField("mua", $("tissueMua").value));
$("tissueMus").addEventListener("input", () => updateTissueField("mus", $("tissueMus").value));
$("tissueG").addEventListener("input", () => updateTissueField("g", $("tissueG").value));
document.querySelectorAll("[data-axis]").forEach((button) => button.addEventListener("click", () => setAxis(button.dataset.axis)));
$("verificationSlice").addEventListener("input", debounce(() => loadAllVerificationSlices().catch((error) => setJobStatus("Slice error", error.message, 1))));

(async function init() {
  try {
    const defaults = await jsonRequest("/api/defaults");
    if (localStorage.getItem("mcvmdl.defaultsRevision") !== defaults.defaultsRevision) {
      ["mcvm.modelPath", "mcvm.tissuePath", "mcvm.numChannels", "mcvm.currentPlanId", "mcvm.currentJobId"].forEach((key) => localStorage.removeItem(key));
      localStorage.setItem("mcvmdl.defaultsRevision", defaults.defaultsRevision);
    }
    $("mcvmPathText").textContent = defaults.mcvmExePath.split(/[\\/]/).pop();
    await loadPlans();
    const jobId = localStorage.getItem("mcvm.currentJobId");
    let storedJob = null;
    if (jobId) {
      try {
        storedJob = await jsonRequest(`/api/mcvm/jobs/${jobId}`);
        const storedCompatible = Boolean(
          storedJob.planId === selectedPlanId()
          && (storedJob.profileVersion || storedJob.profile?.profileVersion) === state.profile?.profileVersion
          && Number(storedJob.mciVersion) === 2
        );
        if (storedCompatible && ["queued", "running", "processing", "cancelling"].includes(storedJob.status)) {
          state.job = storedJob;
          renderProfile(storedJob.profile);
          renderSourceMode(storedJob.sourceSelection || { mode: "planned" });
          $("runMcvmBtn").disabled = true; $("cancelMcvmBtn").disabled = false; pollJob(jobId);
          return;
        }
      } catch (_error) {
        localStorage.removeItem("mcvm.currentJobId");
      }
    }
    // Prefer the newest completed result for the selected plan and profile.
    if (await restoreLatestJobForPlan(selectedPlanId())) return;
    if (storedJob?.status === "completed") {
      state.job = storedJob;
      renderSourceMode(storedJob.sourceSelection || { mode: "planned" });
      renderResults(storedJob);
      setJobStatus("Archived verification loaded", `Loaded legacy result ${storedJob.jobId}.`, 1);
    }
  } catch (error) {
    setJobStatus("Initialization failed", error.message);
  }
})();
