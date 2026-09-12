const $ = (id) => document.getElementById(id);

const SCATTERBRAINS_LABEL_NAMES = {
  0: "Air", 1: "Scalp", 2: "Skull", 6: "CSF", 7: "Gray matter", 8: "White matter",
};

const elements = {
  modelPath: $("modelPath"),
  tissuePath: $("tissuePath"),
  numChannels: $("numChannels"),
  loadTissueBtn: $("loadTissueBtn"),
  runPlanBtn: $("runPlanBtn"),
  rerankBtn: $("rerankBtn"),
  status: $("planStatus"),
  detail: $("planDetail"),
  progress: $("planProgress"),
};

const roiViews = {
  x: { canvas: $("roiCanvasX"), slider: $("roiSliceX"), label: $("roiXLabel"), center: $("centerX") },
  y: { canvas: $("roiCanvasY"), slider: $("roiSliceY"), label: $("roiYLabel"), center: $("centerY") },
  z: { canvas: $("roiCanvasZ"), slider: $("roiSliceZ"), label: $("roiZLabel"), center: $("centerZ") },
};

const bestViews = {
  x: { canvas: $("bestCanvasX"), slider: $("bestSliceX"), label: $("bestXLabel") },
  y: { canvas: $("bestCanvasY"), slider: $("bestSliceY"), label: $("bestYLabel") },
  z: { canvas: $("bestCanvasZ"), slider: $("bestSliceZ"), label: $("bestZLabel") },
};

const state = {
  shape: null,
  tissueInfo: null,
  tissuePreview: null,
  roiImages: new Map(),
  plan: null,
  pollTimer: null,
  roiMetadata: null,
};

function setStatus(title, detail, progress = null) {
  elements.status.textContent = title;
  elements.detail.textContent = detail;
  if (progress !== null) elements.progress.value = Math.max(0, Math.min(1, progress));
}

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ? `${data.error}: ${data.detail}` : data.error || `HTTP ${response.status}`);
  return data;
}

async function postJson(url, payload) {
  return jsonRequest(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function basePayload() {
  return {
    modelPath: elements.modelPath.value.trim(),
    tissuePath: elements.tissuePath.value.trim(),
    numChannels: Number.parseInt(elements.numChannels.value, 10),
  };
}

function saveInputs() {
  localStorage.setItem("mcvm.modelPath", elements.modelPath.value.trim());
  localStorage.setItem("mcvm.tissuePath", elements.tissuePath.value.trim());
  localStorage.setItem("mcvm.numChannels", elements.numChannels.value);
}

function centerValues() {
  return [Number(roiViews.x.center.value), Number(roiViews.y.center.value), Number(roiViews.z.center.value)];
}

function radiusValues() {
  return [Number($("radiusX").value), Number($("radiusY").value), Number($("radiusZ").value)];
}

function weightValues() {
  return {
    target: Number($("targetWeight").value),
    offTarget: Number($("offWeight").value),
    hotspot: Number($("hotspotWeight").value),
  };
}

function selectedTargetLabels() {
  return String($("targetLabels").value || "")
    .split(",")
    .map((value) => Number.parseInt(value.trim(), 10))
    .filter((value) => Number.isInteger(value) && value > 0);
}

function samplingValues() {
  const radiusText = $("patchRadius").value.trim();
  const surfaceText = $("surfaceLabel").value;
  return {
    mode: "localPatch",
    patchRadiusVox: radiusText === "" ? null : Number(radiusText),
    surfaceLabel: surfaceText === "auto" ? "auto" : Number(surfaceText),
    maxIncidenceDeg: Number($("maxIncidence").value),
  };
}

function populateTissueSelectors(labels) {
  const positive = labels.map(Number).filter((label) => label > 0);
  const target = $("targetLabels");
  const previousTarget = target.value;
  target.replaceChildren(new Option("Current center label", ""));
  if (positive.includes(7) && positive.includes(8)) {
    target.add(new Option("Gray + white matter (7, 8)", "7,8"));
  }
  positive.forEach((label) => {
    const name = SCATTERBRAINS_LABEL_NAMES[label] || `Label ${label}`;
    target.add(new Option(`${name} (${label})`, String(label)));
  });
  target.value = previousTarget && [...target.options].some((option) => option.value === previousTarget)
    ? previousTarget
    : (positive.includes(7) && positive.includes(8) ? "7,8" : "");

  const surface = $("surfaceLabel");
  const previousSurface = surface.value;
  surface.replaceChildren(new Option("Auto outer shell", "auto"));
  positive.forEach((label) => {
    const name = SCATTERBRAINS_LABEL_NAMES[label] || `Label ${label}`;
    surface.add(new Option(`${name} (${label})`, String(label)));
  });
  surface.value = [...surface.options].some((option) => option.value === previousSurface) ? previousSurface : "auto";
}

function roiSuggestionPayload(extra = {}) {
  return {
    ...basePayload(),
    roi: { center: centerValues(), radii: radiusValues() },
    targetLabels: selectedTargetLabels(),
    depthRangeVox: [9, 14],
    surfaceLabel: $("surfaceLabel").value,
    ...extra,
  };
}

function showRoiMetadata(metadata) {
  state.roiMetadata = metadata;
  if (!metadata) return;
  $("roiDepthText").textContent = `${Number(metadata.centerDepthVox).toFixed(1)} vox`;
  $("roiCoverText").textContent = `${Number(metadata.coverDepthVox).toFixed(1)} vox`;
  $("roiTissueFraction").textContent = `${(100 * Number(metadata.tissueFraction)).toFixed(1)}%`;
  $("roiTargetFraction").textContent = `${(100 * Number(metadata.targetFraction)).toFixed(1)}%`;
}

async function refreshRoiMetadata() {
  if (!state.shape) return;
  try {
    const result = await postJson("/api/roi-suggestions", roiSuggestionPayload({ inspectOnly: true }));
    showRoiMetadata(result.current);
  } catch (error) {
    ["roiDepthText", "roiCoverText", "roiTissueFraction", "roiTargetFraction"].forEach((id) => { $(id).textContent = "-"; });
  }
}

async function applyRoi(center, radii = null) {
  ["x", "y", "z"].forEach((axis, index) => {
    roiViews[axis].center.value = Math.round(Number(center[index]));
    roiViews[axis].slider.value = roiViews[axis].center.value;
    if (radii) $(`radius${axis.toUpperCase()}`).value = Math.round(Number(radii[index]));
  });
  updateRoiSummary();
  await loadAllRoiSlices();
  await refreshRoiMetadata();
}

async function suggestShallowRoi() {
  const button = $("suggestRoiBtn");
  button.disabled = true;
  setStatus("Finding shallow ROI", "Searching the selected tissue near the external shell.", 0);
  try {
    const result = await postJson("/api/roi-suggestions", roiSuggestionPayload({ count: 1 }));
    const suggestion = result.suggestions?.[0];
    if (!suggestion) throw new Error("No shallow ROI suggestion was returned.");
    await applyRoi(suggestion.center);
    setStatus("Shallow ROI ready", `Center depth ${suggestion.centerDepthVox.toFixed(1)} vox · surface label ${suggestion.surfaceLabel}.`, 0);
  } catch (error) {
    setStatus("ROI suggestion failed", error.message, 0);
  } finally {
    button.disabled = false;
  }
}

async function applyScatterBrainsDemoRoi() {
  const labels = (state.tissueInfo?.labels || []).map(Number);
  if (state.shape?.join(",") !== "128,128,128" || !labels.includes(7) || !labels.includes(8)) {
    setStatus("Preset unavailable", "The ScatterBrains preset requires a 128³ tissue containing labels 7 and 8.", 0);
    return;
  }
  $("targetLabels").value = "7,8";
  await applyRoi([75, 65, 20], [3, 3, 3]);
  setStatus("ScatterBrains shallow preset", "Loaded cortical ROI [75, 65, 20] with radii [3, 3, 3].", 0);
}

function updateRoiSummary() {
  const center = centerValues();
  const radii = radiusValues();
  ["X", "Y", "Z"].forEach((axis, index) => {
    $(`radius${axis}Value`).textContent = radii[index].toFixed(0);
  });
  $("roiCenterText").textContent = center.map((value) => value.toFixed(0)).join(", ");
  $("roiVolumeText").textContent = `${Math.round((4 / 3) * Math.PI * radii[0] * radii[1] * radii[2]).toLocaleString()} voxels`;
}

function planeCoordinates(axis) {
  const center = centerValues();
  const radii = radiusValues();
  if (axis === "x") return { point: [center[2], center[1]], radii: [radii[2], radii[1]] };
  if (axis === "y") return { point: [center[2], center[0]], radii: [radii[2], radii[0]] };
  return { point: [center[1], center[0]], radii: [radii[1], radii[0]] };
}

function drawRoiOverlay(axis) {
  const view = roiViews[axis];
  const image = state.roiImages.get(axis);
  if (!image) return;
  const ctx = view.canvas.getContext("2d", { alpha: false });
  ctx.putImageData(image, 0, 0);
  const { point, radii } = planeCoordinates(axis);
  ctx.save();
  ctx.fillStyle = "rgba(42, 239, 209, 0.12)";
  ctx.strokeStyle = "#2aefd1";
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.ellipse(point[0], point[1], radii[0], radii[1], 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.moveTo(point[0] - 3.5, point[1]);
  ctx.lineTo(point[0] + 3.5, point[1]);
  ctx.moveTo(point[0], point[1] - 3.5);
  ctx.lineTo(point[0], point[1] + 3.5);
  ctx.stroke();
  ctx.restore();
  view.label.textContent = `${axis.toUpperCase()} ${view.slider.value}`;
}

async function loadRgbaPost(url, payload, canvas) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail ? `${error.error}: ${error.detail}` : error.error);
  }
  const width = Number(response.headers.get("X-Width"));
  const height = Number(response.headers.get("X-Height"));
  const bytes = new Uint8ClampedArray(await response.arrayBuffer());
  canvas.width = width;
  canvas.height = height;
  return new ImageData(bytes, width, height);
}

async function loadRoiSlice(axis) {
  if (!state.shape) return;
  const view = roiViews[axis];
  const image = await loadRgbaPost(
    "/api/tissue-slice",
    { ...basePayload(), axis, index: Number(view.slider.value) },
    view.canvas,
  );
  state.roiImages.set(axis, image);
  drawRoiOverlay(axis);
}

async function loadAllRoiSlices() {
  await Promise.all(["x", "y", "z"].map(loadRoiSlice));
}

function configureShape(shape, reset = true) {
  state.shape = shape.map(Number);
  ["x", "y", "z"].forEach((axis, index) => {
    const view = roiViews[axis];
    const maximum = state.shape[index] - 1;
    view.slider.max = maximum;
    view.center.max = maximum;
    bestViews[axis].slider.max = maximum;
    if (reset || Number(view.center.value) > maximum) {
      const middle = Math.floor(state.shape[index] / 2);
      view.center.value = middle;
      view.slider.value = middle;
      bestViews[axis].slider.value = middle;
    }
    const radius = $(`radius${axis.toUpperCase()}`);
    radius.max = Math.max(1, Math.floor(state.shape[index] / 2));
  });
  updateRoiSummary();
}

async function loadTissue() {
  elements.loadTissueBtn.disabled = true;
  setStatus("Loading tissue", "Inspecting labels and preparing orthogonal views…", 0);
  try {
    const info = await postJson("/api/inspect", basePayload());
    if (!info.model.exists) throw new Error(`Weight file not found: ${info.model.path}`);
    if (!info.tissue.exists || !info.tissue.shape) throw new Error(`Tissue file not found: ${info.tissue.path}`);
    state.tissueInfo = info.tissue;
    configureShape(info.tissue.shape, true);
    populateTissueSelectors(info.tissue.labels || []);
    $("inspectText").textContent = `${info.tissue.shape.join(" × ")} voxels · labels ${info.tissue.labels.join(", ")}`;
    saveInputs();
    await Promise.all([
      loadAllRoiSlices(),
      postJson("/api/tissue-preview", { ...basePayload(), maxPoints: 12000 }).then((data) => { state.tissuePreview = data; }),
    ]);
    await refreshRoiMetadata();
    setStatus("Tissue ready", "Click a slice to position the ellipsoidal target ROI.", 0);
  } finally {
    elements.loadTissueBtn.disabled = false;
  }
}

function roiCanvasClick(axis, event) {
  if (!state.shape) return;
  const canvas = roiViews[axis].canvas;
  const rect = canvas.getBoundingClientRect();
  const col = Math.max(0, Math.min(canvas.width - 1, Math.round((event.clientX - rect.left) * canvas.width / rect.width)));
  const row = Math.max(0, Math.min(canvas.height - 1, Math.round((event.clientY - rect.top) * canvas.height / rect.height)));
  if (axis === "x") {
    roiViews.y.center.value = row;
    roiViews.z.center.value = col;
  } else if (axis === "y") {
    roiViews.x.center.value = row;
    roiViews.z.center.value = col;
  } else {
    roiViews.x.center.value = row;
    roiViews.y.center.value = col;
  }
  ["x", "y", "z"].forEach((key) => { roiViews[key].slider.value = roiViews[key].center.value; });
  updateRoiSummary();
  loadAllRoiSlices().catch((error) => setStatus("Slice error", error.message));
  scheduleRoiMetadata();
}

async function upload(kind, file) {
  setStatus("Uploading", `Importing ${file.name}…`);
  const response = await fetch(`/api/upload?kind=${encodeURIComponent(kind)}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
    body: file,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ? `${data.error}: ${data.detail}` : data.error);
  elements[kind === "model" ? "modelPath" : "tissuePath"].value = data.path;
  saveInputs();
  await loadTissue();
}

function planPayload() {
  return {
    ...basePayload(),
    sigma: 10,
    roi: { center: centerValues(), radii: radiusValues() },
    positionCount: Number($("positionCount").value),
    directionCount: Number($("directionCount").value),
    weights: weightValues(),
    sampling: samplingValues(),
  };
}

async function startPlan() {
  if (!state.shape) await loadTissue();
  elements.runPlanBtn.disabled = true;
  elements.rerankBtn.disabled = true;
  $("planningResults").hidden = true;
  try {
    const record = await postJson("/api/plans", planPayload());
    state.plan = record;
    localStorage.setItem("mcvm.currentPlanId", record.planId);
    setStatus("Planning queued", `Plan ${record.planId} is preparing the candidate set.`, 0);
    pollPlan(record.planId);
  } catch (error) {
    elements.runPlanBtn.disabled = false;
    setStatus("Planning failed", error.message, 0);
  }
}

function pollPlan(planId) {
  window.clearTimeout(state.pollTimer);
  const tick = async () => {
    try {
      const record = await jsonRequest(`/api/plans/${encodeURIComponent(planId)}`);
      state.plan = record;
      const completed = Number(record.progress?.completed || 0);
      const total = Number(record.progress?.total || 1);
      setStatus(
        record.status === "running" ? "Screening candidates" : `Planning ${record.status}`,
        record.status === "running" ? `${completed} of ${total} source configurations evaluated.` : `Plan ${planId}`,
        completed / total,
      );
      if (record.status === "completed") {
        elements.runPlanBtn.disabled = false;
        elements.rerankBtn.disabled = false;
        setStatus("Plan completed", `${total} candidates screened in ${record.durationSec.toFixed(2)} seconds.`, 1);
        renderPlan(record);
        return;
      }
      if (["failed", "cancelled"].includes(record.status)) {
        elements.runPlanBtn.disabled = false;
        setStatus("Planning failed", record.error?.detail || record.error?.message || "Unknown planning error", completed / total);
        return;
      }
      state.pollTimer = window.setTimeout(tick, 500);
    } catch (error) {
      elements.runPlanBtn.disabled = false;
      setStatus("Planning status error", error.message);
    }
  };
  tick();
}

function formatMetric(value) {
  if (!Number.isFinite(Number(value))) return "-";
  const numeric = Number(value);
  return numeric === 0 ? "0" : numeric.toExponential(3);
}

function renderTable(record) {
  const body = $("candidateRows");
  body.innerHTML = "";
  record.candidates.slice(0, 5).forEach((candidate) => {
    const row = document.createElement("tr");
    if (candidate.id === record.bestCandidate.id) row.className = "best-row";
    const values = [
      candidate.rank,
      candidate.id,
      formatMetric(candidate.targetAbsorption),
      formatMetric(candidate.offTargetExposure),
      formatMetric(candidate.hotspotRisk),
      Number(candidate.score).toFixed(4),
      candidate.pareto ? "Yes" : "No",
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function setupCanvas(canvas, height = null) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.round(canvas.clientWidth || 600));
  const drawingHeight = height ?? Math.max(220, Math.round(canvas.clientHeight || 240));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(drawingHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height: drawingHeight };
}

function project3d(point, shape, width, height) {
  const normalized = point.map((value, index) => value / Math.max(shape[index] - 1, 1) - 0.5);
  const yaw = -0.72;
  const pitch = 0.42;
  const x1 = normalized[0] * Math.cos(yaw) - normalized[2] * Math.sin(yaw);
  const z1 = normalized[0] * Math.sin(yaw) + normalized[2] * Math.cos(yaw);
  const y1 = normalized[1] * Math.cos(pitch) - z1 * Math.sin(pitch);
  const depth = normalized[1] * Math.sin(pitch) + z1 * Math.cos(pitch);
  const scale = Math.min(width, height) * 0.72;
  return { x: width * 0.5 + x1 * scale, y: height * 0.52 + y1 * scale, depth };
}

function projectLocal(point, anchor, radius, box) {
  const normalized = point.map((value, index) => (Number(value) - Number(anchor[index])) / Math.max(Number(radius), 1));
  const yaw = -0.72;
  const pitch = 0.42;
  const x1 = normalized[0] * Math.cos(yaw) - normalized[2] * Math.sin(yaw);
  const z1 = normalized[0] * Math.sin(yaw) + normalized[2] * Math.cos(yaw);
  const y1 = normalized[1] * Math.cos(pitch) - z1 * Math.sin(pitch);
  const scale = Math.min(box.width, box.height) * 0.34;
  return { x: box.x + box.width * 0.5 + x1 * scale, y: box.y + box.height * 0.56 + y1 * scale };
}

function renderCandidateGeometry(record) {
  const canvas = $("candidateCanvas");
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.fillStyle = "#071415";
  ctx.fillRect(0, 0, width, height);
  if (state.tissuePreview?.points) {
    const points = state.tissuePreview.points;
    const stride = Math.max(1, Math.floor(points.length / 4000));
    ctx.fillStyle = "rgba(154,196,193,.18)";
    for (let index = 0; index < points.length; index += stride) {
      const p = project3d(points[index].slice(0, 3), record.shape, width, height);
      ctx.fillRect(p.x, p.y, 1.2, 1.2);
    }
  }
  const center = record.config.roi.center;
  const roiPoint = project3d(center, record.shape, width, height);
  ctx.strokeStyle = "#2aefd1";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(roiPoint.x, roiPoint.y, 9, 0, Math.PI * 2);
  ctx.stroke();
  record.candidates.forEach((candidate) => {
    const start = project3d(candidate.source, record.shape, width, height);
    const endpoint3d = candidate.source.map((value, index) => value + candidate.direction[index] * 10);
    const end = project3d(endpoint3d, record.shape, width, height);
    const isBest = candidate.id === record.bestCandidate.id;
    ctx.strokeStyle = isBest ? "#ffce54" : candidate.pareto ? "rgba(42,239,209,.72)" : "rgba(167,190,187,.25)";
    ctx.fillStyle = isBest ? "#ff5b54" : candidate.pareto ? "#2aefd1" : "#718d8a";
    ctx.lineWidth = isBest ? 2.5 : 1;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(start.x, start.y, isBest ? 5 : 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
  const sampling = record.sampling || {};
  if (sampling.mode === "localPatch" && sampling.surfaceAnchor) {
    const box = {
      width: Math.min(230, width * 0.38),
      height: Math.min(150, height * 0.68),
      x: width - Math.min(230, width * 0.38) - 12,
      y: 10,
    };
    const radius = Number(sampling.patchRadiusVox || 10);
    ctx.fillStyle = "rgba(4, 24, 25, .94)";
    ctx.strokeStyle = "rgba(121, 192, 187, .46)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(box.x, box.y, box.width, box.height, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "rgba(231,244,242,.86)";
    ctx.font = "10px Segoe UI";
    ctx.fillText(`LOCAL PATCH · ${radius.toFixed(0)} vox`, box.x + 9, box.y + 14);
    const anchorPoint = projectLocal(sampling.surfaceAnchor, sampling.surfaceAnchor, radius, box);
    ctx.strokeStyle = "rgba(255,255,255,.45)";
    ctx.beginPath();
    ctx.arc(anchorPoint.x, anchorPoint.y, Math.min(box.width, box.height) * 0.28, 0, Math.PI * 2);
    ctx.stroke();
    const onePerPosition = record.candidates.filter((candidate) => Number(candidate.directionIndex) === 1);
    onePerPosition.forEach((candidate) => {
      const start = projectLocal(candidate.source, sampling.surfaceAnchor, radius, box);
      const endpoint = candidate.source.map((value, index) => Number(value) + Number(candidate.direction[index]) * radius * 0.42);
      const end = projectLocal(endpoint, sampling.surfaceAnchor, radius, box);
      const isBestSource = Number(candidate.positionIndex) === Number(record.bestCandidate.positionIndex);
      ctx.strokeStyle = isBestSource ? "#ffce54" : "rgba(42,239,209,.58)";
      ctx.fillStyle = isBestSource ? "#ff5b54" : "#2aefd1";
      ctx.lineWidth = isBestSource ? 2 : 0.9;
      ctx.beginPath(); ctx.moveTo(start.x, start.y); ctx.lineTo(end.x, end.y); ctx.stroke();
      ctx.beginPath(); ctx.arc(start.x, start.y, isBestSource ? 3.8 : 2, 0, Math.PI * 2); ctx.fill();
    });
    const localRoi = projectLocal(center, sampling.surfaceAnchor, radius, box);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(localRoi.x, localRoi.y, 5, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.fillStyle = "rgba(231,244,242,.8)";
  ctx.font = "12px Segoe UI";
  ctx.fillText("Surface candidates · arrows show illumination direction", 16, height - 16);
}

function renderPareto(record) {
  const canvas = $("paretoCanvas");
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.fillStyle = "#071415";
  ctx.fillRect(0, 0, width, height);
  const pad = { left: 72, right: 26, top: 38, bottom: 52 };
  const values = record.candidates;
  const xs = values.map((item) => Math.log10(Math.max(item.offTargetExposure, 1e-20)));
  const ys = values.map((item) => Math.log10(Math.max(item.targetAbsorption, 1e-20)));
  const hs = values.map((item) => Math.log10(Math.max(item.hotspotRisk, 1e-20)));
  const range = (array) => {
    let min = Math.min(...array), max = Math.max(...array);
    if (max - min < 1e-9) { min -= 0.5; max += 0.5; }
    return [min, max];
  };
  const [xmin, xmax] = range(xs), [ymin, ymax] = range(ys), [hmin, hmax] = range(hs);
  const px = (value) => pad.left + (value - xmin) / (xmax - xmin) * (width - pad.left - pad.right);
  const py = (value) => height - pad.bottom - (value - ymin) / (ymax - ymin) * (height - pad.top - pad.bottom);
  ctx.strokeStyle = "rgba(192,219,215,.22)";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const x = pad.left + tick / 4 * (width - pad.left - pad.right);
    const y = pad.top + tick / 4 * (height - pad.top - pad.bottom);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, height - pad.bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
  }
  ctx.fillStyle = "rgba(231,244,242,.62)";
  ctx.font = "10px Segoe UI";
  for (let tick = 0; tick <= 4; tick += 1) {
    const x = pad.left + tick / 4 * (width - pad.left - pad.right);
    const y = height - pad.bottom - tick / 4 * (height - pad.top - pad.bottom);
    ctx.textAlign = "center";
    ctx.fillText((xmin + tick / 4 * (xmax - xmin)).toFixed(1), x, height - pad.bottom + 15);
    ctx.textAlign = "right";
    ctx.fillText((ymin + tick / 4 * (ymax - ymin)).toFixed(1), pad.left - 8, y + 3);
  }
  values.forEach((candidate, index) => {
    const hotspot = (hs[index] - hmin) / Math.max(hmax - hmin, 1e-9);
    const red = Math.round(45 + hotspot * 210);
    const green = Math.round(220 - hotspot * 125);
    const isBest = candidate.id === record.bestCandidate.id;
    ctx.fillStyle = `rgb(${red},${green},95)`;
    ctx.strokeStyle = candidate.pareto ? "#ffffff" : "rgba(255,255,255,.18)";
    ctx.lineWidth = candidate.pareto ? 1.5 : 1;
    ctx.beginPath();
    ctx.arc(px(xs[index]), py(ys[index]), isBest ? 7 : 4, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    if (isBest) {
      ctx.strokeStyle = "#ffce54"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px(xs[index]), py(ys[index]), 11, 0, Math.PI * 2); ctx.stroke();
    }
  });
  ctx.font = "10px Segoe UI";
  ctx.textAlign = "left";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.arc(pad.left + 4, 16, 4, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = "rgba(231,244,242,.82)";
  ctx.fillText("Pareto", pad.left + 12, 19);
  ctx.strokeStyle = "#ffce54";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(pad.left + 70, 16, 6, 0, Math.PI * 2); ctx.stroke();
  ctx.fillText("Selected", pad.left + 80, 19);
  const hotspotGradient = ctx.createLinearGradient(width - 150, 0, width - 72, 0);
  hotspotGradient.addColorStop(0, "rgb(45,220,95)");
  hotspotGradient.addColorStop(1, "rgb(255,95,95)");
  ctx.fillStyle = hotspotGradient;
  ctx.fillRect(width - 150, 12, 78, 7);
  ctx.fillStyle = "rgba(231,244,242,.7)";
  ctx.textAlign = "right";
  ctx.fillText("hotspot low → high", width - 26, 28);
  ctx.fillStyle = "rgba(231,244,242,.82)";
  ctx.font = "12px Segoe UI";
  ctx.textAlign = "center";
  ctx.fillText("log10 off-target exposure  → lower is better", (pad.left + width - pad.right) / 2, height - 16);
  ctx.save(); ctx.translate(18, (pad.top + height - pad.bottom) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("log10 target absorption  → higher is better", 0, 0); ctx.restore();
}

function renderBestMetrics(record) {
  const best = record.bestCandidate;
  const items = [
    ["Selected candidate", best.id],
    ["Target absorption", formatMetric(best.targetAbsorption)],
    ["Off-target exposure", formatMetric(best.offTargetExposure)],
    ["Hotspot P99.9", formatMetric(best.hotspotRisk)],
  ];
  const root = $("bestMetrics");
  root.innerHTML = "";
  items.forEach(([label, value]) => {
    const card = document.createElement("div"); card.className = "metric-card";
    const span = document.createElement("span"); span.textContent = label;
    const strong = document.createElement("strong"); strong.textContent = value;
    card.append(span, strong); root.appendChild(card);
  });
}

async function loadBestSlice(axis) {
  if (!state.plan?.planId) return;
  const view = bestViews[axis];
  const index = Number(view.slider.value);
  const response = await fetch(`/api/plans/${state.plan.planId}/slice?axis=${axis}&index=${index}`);
  if (!response.ok) {
    const error = await response.json(); throw new Error(error.detail ? `${error.error}: ${error.detail}` : error.error);
  }
  const width = Number(response.headers.get("X-Width"));
  const height = Number(response.headers.get("X-Height"));
  const bytes = new Uint8ClampedArray(await response.arrayBuffer());
  view.canvas.width = width; view.canvas.height = height;
  view.canvas.getContext("2d", { alpha: false }).putImageData(new ImageData(bytes, width, height), 0, 0);
  view.label.textContent = `${axis.toUpperCase()} ${index}`;
}

function renderPlan(record) {
  state.plan = record;
  const restoredRoi = record.config?.roi;
  if (restoredRoi?.center && restoredRoi?.radii) {
    ["x", "y", "z"].forEach((axis, index) => {
      const center = Math.round(Number(restoredRoi.center[index]));
      roiViews[axis].center.value = center;
      roiViews[axis].slider.value = center;
      $(`radius${axis.toUpperCase()}`).value = Number(restoredRoi.radii[index]);
    });
    updateRoiSummary();
    loadAllRoiSlices().catch((error) => setStatus("Slice error", error.message));
    refreshRoiMetadata();
  }
  if (record.config?.weights) {
    $("targetWeight").value = Number(record.config.weights.target);
    $("offWeight").value = Number(record.config.weights.offTarget);
    $("hotspotWeight").value = Number(record.config.weights.hotspot);
  }
  if (record.config?.positionCount) $("positionCount").value = Number(record.config.positionCount);
  if (record.config?.directionCount) $("directionCount").value = Number(record.config.directionCount);
  $("planningResults").hidden = false;
  renderTable(record);
  renderBestMetrics(record);
  $("durationText").textContent = `${record.candidates.length} candidates · ${record.durationSec.toFixed(2)} s`;
  const sampling = record.sampling || { mode: "legacyGlobal" };
  $("candidateMeta").textContent = sampling.mode === "localPatch"
    ? `${record.config.positionCount} × ${record.config.directionCount} · patch ${Number(sampling.patchRadiusVox).toFixed(0)} vox · shell ${sampling.surfaceLabel}`
    : `${record.config.positionCount} positions × ${record.config.directionCount} directions · legacy global`;
  ["x", "y", "z"].forEach((axis, index) => {
    bestViews[axis].slider.max = record.shape[index] - 1;
    bestViews[axis].slider.value = Math.round(record.config.roi.center[index]);
  });
  renderCandidateGeometry(record);
  renderPareto(record);
  Promise.all(["x", "y", "z"].map(loadBestSlice)).catch((error) => setStatus("Slice error", error.message));
}

async function rerank() {
  if (!state.plan?.planId) return;
  elements.rerankBtn.disabled = true;
  setStatus("Reranking", "Applying the updated balance weights…", 1);
  try {
    const record = await postJson(`/api/plans/${state.plan.planId}/rerank`, { weights: weightValues() });
    renderPlan(record);
    setStatus("Ranking updated", `Selected ${record.bestCandidate.id} from the Pareto front.`, 1);
  } catch (error) {
    setStatus("Reranking failed", error.message, 1);
  } finally {
    elements.rerankBtn.disabled = false;
  }
}

function debounce(fn, delay = 100) {
  let timer;
  return (...args) => { window.clearTimeout(timer); timer = window.setTimeout(() => fn(...args), delay); };
}

const scheduleRoiMetadata = debounce(() => refreshRoiMetadata(), 300);

elements.loadTissueBtn.addEventListener("click", () => loadTissue().catch((error) => setStatus("Load failed", error.message)));
elements.runPlanBtn.addEventListener("click", startPlan);
elements.rerankBtn.addEventListener("click", rerank);
$("suggestRoiBtn").addEventListener("click", suggestShallowRoi);
$("demoRoiBtn").addEventListener("click", () => applyScatterBrainsDemoRoi().catch((error) => setStatus("Preset failed", error.message)));
$("targetLabels").addEventListener("change", scheduleRoiMetadata);
$("surfaceLabel").addEventListener("change", () => {
  const surface = $("surfaceLabel").value === "auto" ? "outer shell" : `label ${$("surfaceLabel").value}`;
  $("samplingSummary").textContent = `${$("patchRadius").value || "Auto"} patch · ${surface} · ≤${$("maxIncidence").value}°`;
  scheduleRoiMetadata();
});
[$("patchRadius"), $("maxIncidence")].forEach((control) => control.addEventListener("input", () => {
  const surface = $("surfaceLabel").value === "auto" ? "outer shell" : `label ${$("surfaceLabel").value}`;
  $("samplingSummary").textContent = `${$("patchRadius").value || "Auto"} patch · ${surface} · ≤${$("maxIncidence").value}°`;
}));
$("uploadModelBtn").addEventListener("click", () => $("modelFileInput").click());
$("uploadTissueBtn").addEventListener("click", () => $("tissueFileInput").click());
$("modelFileInput").addEventListener("change", (event) => { const file = event.target.files?.[0]; if (file) upload("model", file).catch((error) => setStatus("Upload failed", error.message)); event.target.value = ""; });
$("tissueFileInput").addEventListener("change", (event) => { const file = event.target.files?.[0]; if (file) upload("tissue", file).catch((error) => setStatus("Upload failed", error.message)); event.target.value = ""; });

["x", "y", "z"].forEach((axis) => {
  const view = roiViews[axis];
  view.canvas.addEventListener("click", (event) => roiCanvasClick(axis, event));
  view.slider.addEventListener("input", debounce(() => {
    view.center.value = view.slider.value;
    updateRoiSummary();
    loadRoiSlice(axis).catch((error) => setStatus("Slice error", error.message));
    ["x", "y", "z"].filter((key) => key !== axis).forEach(drawRoiOverlay);
    scheduleRoiMetadata();
  }));
  view.center.addEventListener("change", () => {
    view.slider.value = view.center.value;
    updateRoiSummary();
    loadRoiSlice(axis).catch((error) => setStatus("Slice error", error.message));
    ["x", "y", "z"].filter((key) => key !== axis).forEach(drawRoiOverlay);
    scheduleRoiMetadata();
  });
  bestViews[axis].slider.addEventListener("input", debounce(() => loadBestSlice(axis).catch((error) => setStatus("Slice error", error.message))));
});

["X", "Y", "Z"].forEach((axis) => {
  $(`radius${axis}`).addEventListener("input", () => {
    updateRoiSummary();
    ["x", "y", "z"].forEach(drawRoiOverlay);
    scheduleRoiMetadata();
  });
});

window.addEventListener("resize", debounce(() => {
  if (state.plan?.status === "completed") {
    renderCandidateGeometry(state.plan);
    renderPareto(state.plan);
  }
}, 160));

(async function init() {
  try {
    const defaults = await jsonRequest("/api/defaults");
    if (localStorage.getItem("mcvmdl.defaultsRevision") !== defaults.defaultsRevision) {
      ["mcvm.modelPath", "mcvm.tissuePath", "mcvm.numChannels", "mcvm.currentPlanId", "mcvm.currentJobId"].forEach((key) => localStorage.removeItem(key));
      localStorage.setItem("mcvmdl.defaultsRevision", defaults.defaultsRevision);
    }
    elements.modelPath.value = localStorage.getItem("mcvm.modelPath") || defaults.modelPath;
    elements.tissuePath.value = localStorage.getItem("mcvm.tissuePath") || defaults.tissuePath;
    elements.numChannels.value = localStorage.getItem("mcvm.numChannels") || defaults.numChannels;
    $("deviceText").textContent = defaults.device.toUpperCase();
    await loadTissue();
    let planId = localStorage.getItem("mcvm.currentPlanId");
    if (!planId) {
      const data = await jsonRequest("/api/plans");
      planId = data.plans.find((plan) => plan.status === "completed")?.planId || null;
      if (planId) localStorage.setItem("mcvm.currentPlanId", planId);
    }
    if (planId) {
      try {
        const record = await jsonRequest(`/api/plans/${planId}`);
        state.plan = record;
        if (record.status === "completed") {
          elements.rerankBtn.disabled = false;
          renderPlan(record);
          setStatus("Plan restored", `Loaded completed plan ${planId}.`, 1);
        } else if (["queued", "running"].includes(record.status)) {
          elements.runPlanBtn.disabled = true;
          pollPlan(planId);
        }
      } catch (_error) {
        localStorage.removeItem("mcvm.currentPlanId");
      }
    }
  } catch (error) {
    setStatus("Initialization failed", error.message);
  }
})();
