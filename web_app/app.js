const DEFAULT_SIGMA = 10;

const fields = {
  modelPath: document.getElementById("modelPath"),
  tissuePath: document.getElementById("tissuePath"),
  numChannels: document.getElementById("numChannels"),
  sourceX: document.getElementById("sourceX"),
  sourceY: document.getElementById("sourceY"),
  sourceZ: document.getElementById("sourceZ"),
  dirX: document.getElementById("dirX"),
  dirY: document.getElementById("dirY"),
  dirZ: document.getElementById("dirZ"),
};

const sourceSliders = {
  sourceX: document.getElementById("sourceXSlider"),
  sourceY: document.getElementById("sourceYSlider"),
  sourceZ: document.getElementById("sourceZSlider"),
};

const dirSliders = {
  dirX: document.getElementById("dirXSlider"),
  dirY: document.getElementById("dirYSlider"),
  dirZ: document.getElementById("dirZSlider"),
};

const views = {
  x: {
    canvas: document.getElementById("canvasX"),
    slider: document.getElementById("sliderX"),
    label: document.getElementById("xValue"),
  },
  y: {
    canvas: document.getElementById("canvasY"),
    slider: document.getElementById("sliderY"),
    label: document.getElementById("yValue"),
  },
  z: {
    canvas: document.getElementById("canvasZ"),
    slider: document.getElementById("sliderZ"),
    label: document.getElementById("zValue"),
  },
};

const inspectBtn = document.getElementById("inspectBtn");
const simulateBtn = document.getElementById("simulateBtn");
const pickModelBtn = document.getElementById("pickModelBtn");
const pickTissueBtn = document.getElementById("pickTissueBtn");
const browseModelBtn = document.getElementById("browseModelBtn");
const browseTissueBtn = document.getElementById("browseTissueBtn");
const modelFileInput = document.getElementById("modelFileInput");
const tissueFileInput = document.getElementById("tissueFileInput");
const metaLine = document.getElementById("metaLine");
const browserModal = document.getElementById("browserModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalTitle = document.getElementById("modalTitle");
const modalPath = document.getElementById("modalPath");
const modalPathInput = document.getElementById("modalPathInput");
const goPathBtn = document.getElementById("goPathBtn");
const fileList = document.getElementById("fileList");
const tissueCanvas = document.getElementById("tissueCanvas");
const pickerMeta = document.getElementById("pickerMeta");
const pickerSourceText = document.getElementById("pickerSourceText");
const resetViewBtn = document.getElementById("resetViewBtn");
const aimCenterBtn = document.getElementById("aimCenterBtn");

const state = {
  runId: null,
  shape: null,
  predMin: null,
  predMax: null,
  loadingSlices: new Map(),
  browsingKind: null,
  browsingPath: "",
  preview: null,
  previewProjected: [],
  pickerGl: null,
  pickerMesh: null,
  pickerRenderPending: false,
  pickerDrag: null,
  pickerCamera: {
    rotation: quatFromYawPitch(0.12, -0.08),
    zoom: 0.94,
  },
};

const tissuePalette = [
  [36, 44, 49],
  [42, 139, 196],
  [65, 179, 132],
  [237, 183, 67],
  [215, 94, 69],
  [137, 104, 205],
  [74, 181, 197],
  [187, 128, 68],
  [206, 102, 136],
  [124, 151, 81],
  [70, 100, 170],
  [184, 168, 76],
  [92, 150, 142],
  [170, 92, 86],
  [115, 120, 128],
  [54, 166, 96],
  [215, 215, 205],
];

const surfaceMaterial = {
  base: [143, 196, 202],
  shadow: [72, 105, 111],
  highlight: [188, 226, 228],
};

function setStatus(message) {
  metaLine.textContent = message;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function wrapAngle(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const fullTurn = Math.PI * 2;
  return ((((value + Math.PI) % fullTurn) + fullTurn) % fullTurn) - Math.PI;
}

function normalizeQuaternion(q) {
  const norm = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(norm) || norm <= 1e-8) {
    return [0, 0, 0, 1];
  }
  return q.map((value) => value / norm);
}

function multiplyQuaternions(a, b) {
  return normalizeQuaternion([
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ]);
}

function quatFromAxisAngle(axis, angle) {
  const norm = Math.hypot(axis[0], axis[1], axis[2]);
  if (!Number.isFinite(norm) || norm <= 1e-8) {
    return [0, 0, 0, 1];
  }
  const half = angle / 2;
  const scale = Math.sin(half) / norm;
  return normalizeQuaternion([axis[0] * scale, axis[1] * scale, axis[2] * scale, Math.cos(half)]);
}

function quatFromYawPitch(yaw, pitch) {
  const yawQuat = quatFromAxisAngle([0, 1, 0], yaw);
  const pitchQuat = quatFromAxisAngle([1, 0, 0], pitch);
  return multiplyQuaternions(pitchQuat, yawQuat);
}

function quatFromVectors(from, to) {
  const axis = [
    from[1] * to[2] - from[2] * to[1],
    from[2] * to[0] - from[0] * to[2],
    from[0] * to[1] - from[1] * to[0],
  ];
  const dot = clamp(from[0] * to[0] + from[1] * to[1] + from[2] * to[2], -1, 1);
  if (dot < -0.999999) {
    const fallback = Math.abs(from[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0];
    return quatFromAxisAngle(
      [
        from[1] * fallback[2] - from[2] * fallback[1],
        from[2] * fallback[0] - from[0] * fallback[2],
        from[0] * fallback[1] - from[1] * fallback[0],
      ],
      Math.PI,
    );
  }
  return normalizeQuaternion([axis[0], axis[1], axis[2], 1 + dot]);
}

function rotationRowsFromQuaternion(q) {
  const [x, y, z, w] = normalizeQuaternion(q);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  return [
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
  ];
}

function rotateByCamera(value) {
  const rows = rotationRowsFromQuaternion(state.pickerCamera.rotation);
  return {
    x: rows[0][0] * value[0] + rows[0][1] * value[1] + rows[0][2] * value[2],
    y: rows[1][0] * value[0] + rows[1][1] * value[1] + rows[1][2] * value[2],
    depth: rows[2][0] * value[0] + rows[2][1] * value[1] + rows[2][2] * value[2],
  };
}

function arcballVector(event) {
  const rect = tissueCanvas.getBoundingClientRect();
  const size = Math.max(Math.min(rect.width, rect.height), 1);
  const x = (2 * (event.clientX - rect.left) - rect.width) / size;
  const y = (rect.height - 2 * (event.clientY - rect.top)) / size;
  const lengthSquared = x * x + y * y;
  if (lengthSquared <= 1) {
    return [x, y, Math.sqrt(1 - lengthSquared)];
  }
  const length = Math.sqrt(lengthSquared);
  return [x / length, y / length, 0];
}

function numberValue(id) {
  const value = Number(fields[id].value);
  if (!Number.isFinite(value)) {
    throw new Error(`${id} is not a valid number`);
  }
  return value;
}

function currentSource() {
  return [Number(fields.sourceX.value), Number(fields.sourceY.value), Number(fields.sourceZ.value)];
}

function currentDirection() {
  return [Number(fields.dirX.value), Number(fields.dirY.value), Number(fields.dirZ.value)];
}

function normalizeVector(vector) {
  const norm = Math.hypot(vector[0], vector[1], vector[2]);
  if (!Number.isFinite(norm) || norm <= 1e-8) {
    return [0, 0, 1];
  }
  return vector.map((value) => value / norm);
}

function formatCompact(value, digits = 1) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return Number(value).toFixed(digits).replace(/\.0+$/, "");
}

function setSourceValues(source) {
  const keys = ["sourceX", "sourceY", "sourceZ"];
  keys.forEach((key, index) => {
    const value = Number(source[index]);
    if (!Number.isFinite(value)) {
      return;
    }
    fields[key].value = value.toFixed(1);
    sourceSliders[key].value = String(value);
  });
  updatePickerSourceText();
  requestPickerRender();
}

function setDirectionValues(direction) {
  const normalized = normalizeVector(direction.map(Number));
  const keys = ["dirX", "dirY", "dirZ"];
  keys.forEach((key, index) => {
    fields[key].value = normalized[index].toFixed(2);
    dirSliders[key].value = String(normalized[index]);
  });
  requestPickerRender();
}

function sourceText(source = currentSource()) {
  return `Source ${source.map((value) => formatCompact(value, 1)).join(", ")}`;
}

function updatePickerSourceText() {
  if (pickerSourceText) {
    pickerSourceText.textContent = sourceText();
  }
}

function payloadBase() {
  return {
    modelPath: fields.modelPath.value.trim(),
    tissuePath: fields.tissuePath.value.trim(),
    numChannels: Number.parseInt(fields.numChannels.value, 10),
  };
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail ? `${data.error}: ${data.detail}` : data.error);
  }
  return data;
}

function setSliders(shape) {
  const axes = ["x", "y", "z"];
  axes.forEach((axis, idx) => {
    const slider = views[axis].slider;
    const middle = Math.floor(shape[idx] / 2);
    slider.min = "0";
    slider.max = String(shape[idx] - 1);
    slider.value = String(middle);
    slider.disabled = !state.runId;
    views[axis].label.textContent = `${axis.toUpperCase()} ${middle}`;
  });
  setSourceRanges(shape, false);
}

function setSourceRanges(shape, resetToMiddle = false) {
  const keys = ["sourceX", "sourceY", "sourceZ"];
  keys.forEach((key, idx) => {
    const slider = sourceSliders[key];
    const max = Math.max(0, Number(shape[idx]) - 1);
    slider.min = "0";
    slider.max = String(max);
    slider.step = "0.1";
    if (resetToMiddle) {
      const middle = Math.floor(Number(shape[idx]) / 2);
      fields[key].value = String(middle);
      slider.value = String(middle);
    } else {
      const current = Math.min(Math.max(Number(fields[key].value || 0), 0), max);
      fields[key].value = String(current);
      slider.value = String(current);
    }
  });
  updatePickerSourceText();
  requestPickerRender();
}

function bindLinkedNumberAndSlider(input, slider, formatter = (v) => String(v), onChange = null) {
  input.addEventListener("input", () => {
    const value = Number(input.value);
    if (Number.isFinite(value)) {
      slider.value = String(value);
      if (onChange) {
        onChange();
      }
    }
  });
  slider.addEventListener("input", () => {
    input.value = formatter(Number(slider.value));
    if (onChange) {
      onChange();
    }
  });
}

function updateInspectReadout(data) {
  const tissue = data.tissue;
  if (!tissue.exists) {
    setStatus("Tissue file not found");
    state.preview = null;
    pickerMeta.textContent = "No tissue preview";
    requestPickerRender();
    return;
  }
  const shape = tissue.shape || [];
  setStatus(`${shape.join(" x ")} volume, ${tissue.dtype}, labels ${tissue.min}..${tissue.max}, ${tissue.labelCount} classes`);
  if (shape.length === 3) {
    state.shape = shape;
    setSliders(shape);
  }
}

async function loadTissuePreview() {
  if (!fields.tissuePath.value.trim()) {
    return;
  }
  pickerMeta.textContent = "Loading surface mesh";
  const data = await postJson("/api/tissue-mesh", {
    ...payloadBase(),
    stride: 1,
    maxFaces: 90000,
  });
  state.preview = {
    shape: data.shape,
    faces: data.faces,
    faceCount: data.faceCount,
    stride: data.stride,
  };
  buildPickerMeshBuffers();
  pickerMeta.textContent = `${data.faceCount.toLocaleString()} solid surface faces, stride ${data.stride}`;
  requestPickerRender();
}

function canvasMetrics(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, dpr };
}

function projectVoxel(voxel, shape, metrics) {
  const center = [(shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2];
  const scale = Math.max(shape[0], shape[1], shape[2]) / 2 || 1;
  const x = (Number(voxel[0]) - center[0]) / scale;
  const y = (Number(voxel[1]) - center[1]) / scale;
  const z = (Number(voxel[2]) - center[2]) / scale;
  const rotated = rotateByCamera([x, y, z]);
  const viewScale = Math.min(metrics.width, metrics.height) * 0.36 * state.pickerCamera.zoom;
  return {
    x: metrics.width / 2 + rotated.x * viewScale,
    y: metrics.height / 2 + rotated.y * viewScale,
    depth: rotated.depth,
  };
}

function rotateVector(vector) {
  return rotateByCamera([Number(vector[0]), Number(vector[1]), Number(vector[2])]);
}

function polygonContainsPoint(points, x, y) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const xi = points[i].x;
    const yi = points[i].y;
    const xj = points[j].x;
    const yj = points[j].y;
    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-8) + xi;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function mixColor(from, to, amount) {
  const t = clamp(amount, 0, 1);
  return [
    Math.round(from[0] + (to[0] - from[0]) * t),
    Math.round(from[1] + (to[1] - from[1]) * t),
    Math.round(from[2] + (to[2] - from[2]) * t),
  ];
}

function surfaceRgb(shade) {
  if (shade < 0.78) {
    return mixColor(surfaceMaterial.shadow, surfaceMaterial.base, (shade - 0.44) / 0.34);
  }
  return mixColor(surfaceMaterial.base, surfaceMaterial.highlight, (shade - 0.78) / 0.34);
}

function makeShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const detail = gl.getShaderInfoLog(shader) || "Unknown shader error";
    gl.deleteShader(shader);
    throw new Error(detail);
  }
  return shader;
}

function makeProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, makeShader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, makeShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const detail = gl.getProgramInfoLog(program) || "Unknown WebGL link error";
    gl.deleteProgram(program);
    throw new Error(detail);
  }
  return program;
}

function createPickerRenderer() {
  if (state.pickerGl) {
    return state.pickerGl;
  }
  const webglOptions = { antialias: true, alpha: false, preserveDrawingBuffer: true };
  const gl = tissueCanvas.getContext("webgl", webglOptions) || tissueCanvas.getContext("experimental-webgl", webglOptions);
  if (!gl) {
    throw new Error("WebGL is unavailable in this browser");
  }

  const sharedProjection = `
    precision mediump float;
    uniform vec3 u_center;
    uniform float u_scale;
    uniform vec3 u_rot0;
    uniform vec3 u_rot1;
    uniform vec3 u_rot2;
    uniform float u_zoom;
    uniform float u_aspect;
    vec3 rotateCamera(vec3 value) {
      return vec3(dot(u_rot0, value), dot(u_rot1, value), dot(u_rot2, value));
    }
    vec4 projectVoxel(vec3 value) {
      vec3 p = rotateCamera((value - u_center) / u_scale);
      return vec4(p.x * u_zoom / u_aspect, -p.y * u_zoom, -p.z * 0.55, 1.0);
    }
  `;

  const meshVertex = `
    ${sharedProjection}
    attribute vec3 a_position;
    attribute vec3 a_normal;
    varying vec3 v_normal;
    varying float v_depth;
    void main() {
      vec3 p = rotateCamera((a_position - u_center) / u_scale);
      v_normal = normalize(rotateCamera(a_normal));
      v_depth = p.z;
      gl_Position = vec4(p.x * u_zoom / u_aspect, -p.y * u_zoom, -p.z * 0.55, 1.0);
    }
  `;
  const meshFragment = `
    precision mediump float;
    uniform vec3 u_base;
    uniform vec3 u_shadow;
    uniform vec3 u_highlight;
    varying vec3 v_normal;
    varying float v_depth;
    void main() {
      vec3 normal = normalize(v_normal);
      vec3 light = normalize(vec3(-0.52, 0.36, 0.78));
      float lit = max(dot(normal, light), 0.0);
      float side = max(dot(normal, normalize(vec3(0.78, -0.18, 0.62))), 0.0);
      float shade = clamp(0.48 + lit * 0.42 + side * 0.10 + v_depth * 0.055, 0.0, 1.12);
      vec3 color = mix(u_shadow, u_base, clamp(shade, 0.0, 1.0));
      color = mix(color, u_highlight, clamp((shade - 0.86) * 0.35, 0.0, 0.18));
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  const simpleVertex = `
    ${sharedProjection}
    attribute vec3 a_position;
    uniform float u_pointSize;
    void main() {
      gl_Position = projectVoxel(a_position);
      gl_PointSize = u_pointSize;
    }
  `;
  const simpleFragment = `
    precision mediump float;
    uniform vec4 u_color;
    void main() {
      gl_FragColor = u_color;
    }
  `;

  const renderer = {
    gl,
    meshProgram: makeProgram(gl, meshVertex, meshFragment),
    simpleProgram: makeProgram(gl, simpleVertex, simpleFragment),
    meshPositionBuffer: gl.createBuffer(),
    meshNormalBuffer: gl.createBuffer(),
    meshEdgeBuffer: gl.createBuffer(),
    lineBuffer: gl.createBuffer(),
    meshVertexCount: 0,
    meshEdgeVertexCount: 0,
  };
  renderer.meshLocations = {
    position: gl.getAttribLocation(renderer.meshProgram, "a_position"),
    normal: gl.getAttribLocation(renderer.meshProgram, "a_normal"),
    center: gl.getUniformLocation(renderer.meshProgram, "u_center"),
    scale: gl.getUniformLocation(renderer.meshProgram, "u_scale"),
    rot0: gl.getUniformLocation(renderer.meshProgram, "u_rot0"),
    rot1: gl.getUniformLocation(renderer.meshProgram, "u_rot1"),
    rot2: gl.getUniformLocation(renderer.meshProgram, "u_rot2"),
    zoom: gl.getUniformLocation(renderer.meshProgram, "u_zoom"),
    aspect: gl.getUniformLocation(renderer.meshProgram, "u_aspect"),
    base: gl.getUniformLocation(renderer.meshProgram, "u_base"),
    shadow: gl.getUniformLocation(renderer.meshProgram, "u_shadow"),
    highlight: gl.getUniformLocation(renderer.meshProgram, "u_highlight"),
  };
  renderer.simpleLocations = {
    position: gl.getAttribLocation(renderer.simpleProgram, "a_position"),
    center: gl.getUniformLocation(renderer.simpleProgram, "u_center"),
    scale: gl.getUniformLocation(renderer.simpleProgram, "u_scale"),
    rot0: gl.getUniformLocation(renderer.simpleProgram, "u_rot0"),
    rot1: gl.getUniformLocation(renderer.simpleProgram, "u_rot1"),
    rot2: gl.getUniformLocation(renderer.simpleProgram, "u_rot2"),
    zoom: gl.getUniformLocation(renderer.simpleProgram, "u_zoom"),
    aspect: gl.getUniformLocation(renderer.simpleProgram, "u_aspect"),
    color: gl.getUniformLocation(renderer.simpleProgram, "u_color"),
    pointSize: gl.getUniformLocation(renderer.simpleProgram, "u_pointSize"),
  };
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  state.pickerGl = renderer;
  return renderer;
}

function setPickerUniforms(renderer, program, locations, shape, metrics) {
  const gl = renderer.gl;
  const center = [(shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2];
  const scale = Math.max(shape[0], shape[1], shape[2]) / 2 || 1;
  gl.useProgram(program);
  gl.uniform3f(locations.center, center[0], center[1], center[2]);
  gl.uniform1f(locations.scale, scale);
  const rows = rotationRowsFromQuaternion(state.pickerCamera.rotation);
  gl.uniform3f(locations.rot0, rows[0][0], rows[0][1], rows[0][2]);
  gl.uniform3f(locations.rot1, rows[1][0], rows[1][1], rows[1][2]);
  gl.uniform3f(locations.rot2, rows[2][0], rows[2][1], rows[2][2]);
  gl.uniform1f(locations.zoom, state.pickerCamera.zoom);
  gl.uniform1f(locations.aspect, metrics.width / Math.max(metrics.height, 1));
}

function buildPickerMeshBuffers() {
  if (!state.preview?.faces?.length) {
    state.pickerMesh = null;
    return;
  }
  const renderer = createPickerRenderer();
  const gl = renderer.gl;
  const vertexCount = state.preview.faces.length * 6;
  const edgeVertexCount = state.preview.faces.length * 8;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const edges = new Float32Array(edgeVertexCount * 3);
  let offset = 0;
  let edgeOffset = 0;
  const order = [0, 1, 2, 0, 2, 3];
  const edgeOrder = [0, 1, 1, 2, 2, 3, 3, 0];
  state.preview.faces.forEach((face) => {
    order.forEach((vertexIndex) => {
      const vertex = face.v[vertexIndex];
      positions[offset] = Number(vertex[0]);
      positions[offset + 1] = Number(vertex[1]);
      positions[offset + 2] = Number(vertex[2]);
      normals[offset] = Number(face.n[0]);
      normals[offset + 1] = Number(face.n[1]);
      normals[offset + 2] = Number(face.n[2]);
      offset += 3;
    });
    edgeOrder.forEach((vertexIndex) => {
      const vertex = face.v[vertexIndex];
      edges[edgeOffset] = Number(vertex[0]);
      edges[edgeOffset + 1] = Number(vertex[1]);
      edges[edgeOffset + 2] = Number(vertex[2]);
      edgeOffset += 3;
    });
  });
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.meshPositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.meshNormalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.meshEdgeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, edges, gl.STATIC_DRAW);
  renderer.meshVertexCount = vertexCount;
  renderer.meshEdgeVertexCount = edgeVertexCount;
  state.pickerMesh = {
    faceCount: state.preview.faces.length,
    stride: state.preview.stride,
  };
}

function drawBounds(ctx, shape, metrics) {
  const maxX = shape[0] - 1;
  const maxY = shape[1] - 1;
  const maxZ = shape[2] - 1;
  const corners = [
    [0, 0, 0],
    [maxX, 0, 0],
    [maxX, maxY, 0],
    [0, maxY, 0],
    [0, 0, maxZ],
    [maxX, 0, maxZ],
    [maxX, maxY, maxZ],
    [0, maxY, maxZ],
  ].map((corner) => projectVoxel(corner, shape, metrics));
  const edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  ctx.strokeStyle = "rgba(20, 34, 30, 0.24)";
  ctx.lineWidth = Math.max(1, metrics.dpr);
  ctx.beginPath();
  edges.forEach(([a, b]) => {
    ctx.moveTo(corners[a].x, corners[a].y);
    ctx.lineTo(corners[b].x, corners[b].y);
  });
  ctx.stroke();
}

function drawDirectionArrow(ctx, shape, metrics) {
  const source = currentSource();
  if (!source.every(Number.isFinite)) {
    return;
  }
  const direction = normalizeVector(currentDirection());
  const span = Math.max(shape[0], shape[1], shape[2]) * 0.18;
  const tip = [
    source[0] + direction[0] * span,
    source[1] + direction[1] * span,
    source[2] + direction[2] * span,
  ];
  const start = projectVoxel(source, shape, metrics);
  const end = projectVoxel(tip, shape, metrics);
  ctx.strokeStyle = "rgba(255, 244, 168, 0.92)";
  ctx.lineWidth = 2 * metrics.dpr;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  ctx.fillStyle = "#e22a2a";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2 * metrics.dpr;
  ctx.beginPath();
  ctx.arc(start.x, start.y, 6 * metrics.dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function boundsVertices(shape) {
  const maxX = shape[0] - 1;
  const maxY = shape[1] - 1;
  const maxZ = shape[2] - 1;
  const corners = [
    [0, 0, 0],
    [maxX, 0, 0],
    [maxX, maxY, 0],
    [0, maxY, 0],
    [0, 0, maxZ],
    [maxX, 0, maxZ],
    [maxX, maxY, maxZ],
    [0, maxY, maxZ],
  ];
  const edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  return edges.flatMap(([a, b]) => [...corners[a], ...corners[b]]);
}

function drawSimpleVertices(renderer, shape, metrics, vertices, mode, color, pointSize = 1) {
  if (!vertices.length) {
    return;
  }
  const gl = renderer.gl;
  setPickerUniforms(renderer, renderer.simpleProgram, renderer.simpleLocations, shape, metrics);
  gl.uniform4f(renderer.simpleLocations.color, color[0], color[1], color[2], color[3] ?? 1);
  gl.uniform1f(renderer.simpleLocations.pointSize, pointSize);
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.lineBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(renderer.simpleLocations.position);
  gl.vertexAttribPointer(renderer.simpleLocations.position, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(mode, 0, vertices.length / 3);
}

function sourceArrowVertices(shape) {
  const source = currentSource();
  if (!source.every(Number.isFinite)) {
    return { line: [], point: [] };
  }
  const direction = normalizeVector(currentDirection());
  const span = Math.max(shape[0], shape[1], shape[2]) * 0.18;
  const tip = [
    source[0] + direction[0] * span,
    source[1] + direction[1] * span,
    source[2] + direction[2] * span,
  ];
  return {
    line: [...source, ...tip],
    point: source,
  };
}

function renderTissuePreview() {
  if (!tissueCanvas) {
    return;
  }
  let renderer;
  try {
    renderer = createPickerRenderer();
  } catch (error) {
    pickerMeta.textContent = error.message;
    return;
  }
  const metrics = canvasMetrics(tissueCanvas);
  const gl = renderer.gl;
  gl.viewport(0, 0, metrics.width, metrics.height);
  gl.clearColor(0.969, 0.98, 0.973, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (!state.preview || !state.preview.faces.length || !state.pickerMesh) {
    updatePickerSourceText();
    return;
  }

  const shape = state.preview.shape;
  gl.disable(gl.DEPTH_TEST);
  drawSimpleVertices(renderer, shape, metrics, boundsVertices(shape), gl.LINES, [0.52, 0.6, 0.58], 1);

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  setPickerUniforms(renderer, renderer.meshProgram, renderer.meshLocations, shape, metrics);
  gl.uniform3f(renderer.meshLocations.base, surfaceMaterial.base[0] / 255, surfaceMaterial.base[1] / 255, surfaceMaterial.base[2] / 255);
  gl.uniform3f(renderer.meshLocations.shadow, surfaceMaterial.shadow[0] / 255, surfaceMaterial.shadow[1] / 255, surfaceMaterial.shadow[2] / 255);
  gl.uniform3f(renderer.meshLocations.highlight, surfaceMaterial.highlight[0] / 255, surfaceMaterial.highlight[1] / 255, surfaceMaterial.highlight[2] / 255);
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.meshPositionBuffer);
  gl.enableVertexAttribArray(renderer.meshLocations.position);
  gl.vertexAttribPointer(renderer.meshLocations.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.meshNormalBuffer);
  gl.enableVertexAttribArray(renderer.meshLocations.normal);
  gl.vertexAttribPointer(renderer.meshLocations.normal, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLES, 0, renderer.meshVertexCount);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  setPickerUniforms(renderer, renderer.simpleProgram, renderer.simpleLocations, shape, metrics);
  gl.uniform4f(renderer.simpleLocations.color, 0.16, 0.34, 0.37, 0.32);
  gl.uniform1f(renderer.simpleLocations.pointSize, 1);
  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.meshEdgeBuffer);
  gl.enableVertexAttribArray(renderer.simpleLocations.position);
  gl.vertexAttribPointer(renderer.simpleLocations.position, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINES, 0, renderer.meshEdgeVertexCount);
  gl.disable(gl.BLEND);

  const source = sourceArrowVertices(shape);
  gl.disable(gl.DEPTH_TEST);
  drawSimpleVertices(renderer, shape, metrics, source.line, gl.LINES, [1.0, 0.92, 0.42], 1);
  drawSimpleVertices(renderer, shape, metrics, source.point, gl.POINTS, [1.0, 1.0, 1.0], 15 * metrics.dpr);
  drawSimpleVertices(renderer, shape, metrics, source.point, gl.POINTS, [0.88, 0.08, 0.12], 10 * metrics.dpr);
  updatePickerSourceText();
}

function requestPickerRender() {
  if (state.pickerRenderPending) {
    return;
  }
  state.pickerRenderPending = true;
  window.requestAnimationFrame(() => {
    state.pickerRenderPending = false;
    renderTissuePreview();
  });
}

async function loadDefaults() {
  const response = await fetch("/api/defaults");
  const defaults = await response.json();
  if (localStorage.getItem("mcvmdl.defaultsRevision") !== defaults.defaultsRevision) {
    ["mcvm.modelPath", "mcvm.tissuePath", "mcvm.numChannels", "mcvm.currentPlanId", "mcvm.currentJobId"].forEach((key) => localStorage.removeItem(key));
    localStorage.setItem("mcvmdl.defaultsRevision", defaults.defaultsRevision);
  }
  fields.modelPath.value = localStorage.getItem("mcvm.modelPath") || defaults.modelPath;
  fields.tissuePath.value = localStorage.getItem("mcvm.tissuePath") || defaults.tissuePath;
  fields.numChannels.value = localStorage.getItem("mcvm.numChannels") || defaults.numChannels;
  setSourceValues(defaults.source);
  setDirectionValues(defaults.direction);
}

function saveSharedSettings() {
  localStorage.setItem("mcvm.modelPath", fields.modelPath.value.trim());
  localStorage.setItem("mcvm.tissuePath", fields.tissuePath.value.trim());
  localStorage.setItem("mcvm.numChannels", fields.numChannels.value);
}

async function inspect() {
  setStatus("Inspecting model and tissue files...");
  inspectBtn.disabled = true;
  try {
    saveSharedSettings();
    const data = await postJson("/api/inspect", payloadBase());
    updateInspectReadout(data);
    const modelText = data.model.exists ? "model ready" : "model missing";
    const tissueText = data.tissue.exists ? "tissue ready" : "tissue missing";
    const channelText = data.tissue.labelsWithinChannels === false ? "labels exceed channel count" : "labels in range";
    setStatus(`Inspect complete: ${modelText}, ${tissueText}, ${channelText}`);
    if (data.tissue.exists && Array.isArray(data.tissue.shape) && data.tissue.shape.length === 3) {
      try {
        await loadTissuePreview();
      } catch (error) {
        pickerMeta.textContent = error.message;
        state.preview = null;
        requestPickerRender();
      }
    }
  } catch (error) {
    setStatus(error.message);
  } finally {
    inspectBtn.disabled = false;
  }
}

async function simulate() {
  setStatus("Running model inference...");
  simulateBtn.disabled = true;
  inspectBtn.disabled = true;
  try {
    const payload = {
      ...payloadBase(),
      source: [numberValue("sourceX"), numberValue("sourceY"), numberValue("sourceZ")],
      direction: [numberValue("dirX"), numberValue("dirY"), numberValue("dirZ")],
      sigma: DEFAULT_SIGMA,
    };
    const data = await postJson("/api/simulate", payload);
    state.runId = data.runId;
    state.shape = data.shape;
    state.predMin = data.prediction.min;
    state.predMax = data.prediction.max;
    setSliders(data.shape);
    setStatus(`Run ${data.runId}: ${data.shape.join(" x ")}, ${data.inferenceSec.toFixed(3)} s, range ${data.prediction.min.toFixed(5)}..${data.prediction.max.toFixed(5)}`);
    await loadAllSlices();
  } catch (error) {
    setStatus(error.message);
  } finally {
    simulateBtn.disabled = false;
    inspectBtn.disabled = false;
  }
}

async function loadSlice(axis) {
  if (!state.runId) {
    return;
  }
  const view = views[axis];
  const index = Number.parseInt(view.slider.value, 10);
  view.label.textContent = `${axis.toUpperCase()} ${index}`;
  const key = `${axis}:${index}`;
  state.loadingSlices.set(axis, key);
  const response = await fetch(`/api/slice?runId=${encodeURIComponent(state.runId)}&axis=${axis}&index=${index}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail ? `${error.error}: ${error.detail}` : error.error);
  }
  if (state.loadingSlices.get(axis) !== key) {
    return;
  }
  const width = Number(response.headers.get("X-Width"));
  const height = Number(response.headers.get("X-Height"));
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8ClampedArray(buffer);
  if (bytes.length !== width * height * 4) {
    throw new Error(`Unexpected slice byte length: ${bytes.length}`);
  }
  const canvas = view.canvas;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.putImageData(new ImageData(bytes, width, height), 0, 0);
}

async function loadAllSlices() {
  await Promise.all(["x", "y", "z"].map((axis) => loadSlice(axis)));
}

function debounce(fn, delay = 80) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

const debouncedSliceLoad = debounce((axis) => {
  loadSlice(axis).catch((error) => setStatus(error.message));
});

function setPathForKind(kind, path) {
  if (kind === "model") {
    fields.modelPath.value = path;
  } else if (kind === "tissue") {
    fields.tissuePath.value = path;
  }
}

function currentPathForKind(kind) {
  return kind === "model" ? fields.modelPath.value.trim() : fields.tissuePath.value.trim();
}

async function uploadFile(kind, file) {
  setStatus(`Importing ${file.name}...`);
  const response = await fetch(`/api/upload?kind=${encodeURIComponent(kind)}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Filename": encodeURIComponent(file.name),
    },
    body: file,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail ? `${data.error}: ${data.detail}` : data.error);
  }
  setPathForKind(kind, data.path);
  setStatus(`Imported ${data.name}`);
  await inspect();
}

function setupDropTarget(textarea, kind) {
  textarea.addEventListener("dragover", (event) => {
    event.preventDefault();
    textarea.closest(".path-block")?.classList.add("dragging");
  });
  textarea.addEventListener("dragleave", () => {
    textarea.closest(".path-block")?.classList.remove("dragging");
  });
  textarea.addEventListener("drop", async (event) => {
    event.preventDefault();
    textarea.closest(".path-block")?.classList.remove("dragging");
    try {
      const file = event.dataTransfer.files?.[0];
      if (file) {
        await uploadFile(kind, file);
        return;
      }
      const text = event.dataTransfer.getData("text/plain");
      if (text) {
        setPathForKind(kind, text.trim().replace(/^file:[/][/]/i, ""));
        await inspect();
      }
    } catch (error) {
      setStatus(error.message);
    }
  });
}

async function openBrowser(kind, path = "") {
  state.browsingKind = kind;
  state.browsingPath = path || currentPathForKind(kind);
  browserModal.hidden = false;
  modalTitle.textContent = kind === "model" ? "Select Weight File" : "Select Tissue File";
  await loadBrowserPath(state.browsingKind, state.browsingPath);
}

async function loadBrowserPath(kind, path) {
  const url = `/api/list-path?kind=${encodeURIComponent(kind)}&path=${encodeURIComponent(path || "")}`;
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail ? `${data.error}: ${data.detail}` : data.error);
  }
  state.browsingPath = data.path;
  modalPath.textContent = data.path;
  modalPathInput.value = data.path;
  fileList.innerHTML = "";
  if (data.parent) {
    fileList.appendChild(fileRow({ name: "..", path: data.parent, isDir: true, selectable: false }, kind));
  }
  if (!data.entries.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No selectable files in this directory.";
    fileList.appendChild(empty);
    return;
  }
  data.entries.forEach((entry) => fileList.appendChild(fileRow(entry, kind)));
}

function fileRow(entry, kind) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = `file-row ${entry.isDir ? "is-dir" : "is-file"}`;
  row.innerHTML = `<span class="file-icon">${entry.isDir ? "DIR" : "FILE"}</span><strong></strong><small></small>`;
  row.querySelector("strong").textContent = entry.name;
  row.querySelector("small").textContent = entry.isDir ? "Folder" : formatBytes(entry.size);
  row.addEventListener("click", async () => {
    try {
      if (entry.isDir) {
        await loadBrowserPath(kind, entry.path);
      } else if (entry.selectable) {
        setPathForKind(kind, entry.path);
        browserModal.hidden = true;
        await inspect();
      }
    } catch (error) {
      setStatus(error.message);
    }
  });
  return row;
}

function formatBytes(size) {
  if (!Number.isFinite(size)) {
    return "";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function aimDirectionAtCenter() {
  const shape = state.preview?.shape || state.shape;
  if (!shape || shape.length !== 3) {
    return;
  }
  const source = currentSource();
  if (!source.every(Number.isFinite)) {
    return;
  }
  const center = [(shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2];
  setDirectionValues(center.map((value, index) => value - source[index]));
}

function projectedFacesForPicking() {
  if (!state.preview?.faces?.length) {
    return [];
  }
  const metrics = canvasMetrics(tissueCanvas);
  const shape = state.preview.shape;
  const projected = state.preview.faces.map((face) => {
    const points = face.v.map((vertex) => projectVoxel(vertex, shape, metrics));
    const depth = points.reduce((sum, point) => sum + point.depth, 0) / points.length;
    const source = [
      face.v.reduce((sum, vertex) => sum + Number(vertex[0]), 0) / face.v.length,
      face.v.reduce((sum, vertex) => sum + Number(vertex[1]), 0) / face.v.length,
      face.v.reduce((sum, vertex) => sum + Number(vertex[2]), 0) / face.v.length,
    ];
    return { points, depth, source };
  });
  projected.sort((a, b) => a.depth - b.depth);
  return projected;
}

function selectPickerPoint(event) {
  if (!state.preview?.faces?.length) {
    return;
  }
  const rect = tissueCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }
  const x = (event.clientX - rect.left) * (tissueCanvas.width / rect.width);
  const y = (event.clientY - rect.top) * (tissueCanvas.height / rect.height);
  const projectedFaces = projectedFacesForPicking();
  let best = null;
  for (let index = projectedFaces.length - 1; index >= 0; index -= 1) {
    const item = projectedFaces[index];
    if (polygonContainsPoint(item.points, x, y)) {
      best = item;
      break;
    }
  }
  if (!best) {
    const radius = 18 * (tissueCanvas.width / rect.width);
    let bestDistance = radius * radius;
    for (let index = projectedFaces.length - 1; index >= 0; index -= 1) {
      const item = projectedFaces[index];
      const centerX = item.points.reduce((sum, point) => sum + point.x, 0) / item.points.length;
      const centerY = item.points.reduce((sum, point) => sum + point.y, 0) / item.points.length;
      const dx = centerX - x;
      const dy = centerY - y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = item;
      }
    }
  }
  if (!best) {
    return;
  }
  setSourceValues(best.source);
  aimDirectionAtCenter();
}

function resetPickerView() {
  state.pickerCamera.rotation = quatFromYawPitch(0.12, -0.08);
  state.pickerCamera.zoom = 0.94;
  requestPickerRender();
}

function pickerPointerDown(event) {
  if (!state.preview) {
    return;
  }
  state.pickerDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    startVector: arcballVector(event),
    startRotation: state.pickerCamera.rotation.slice(),
    moved: false,
  };
  tissueCanvas.setPointerCapture(event.pointerId);
}

function pickerPointerMove(event) {
  const drag = state.pickerDrag;
  if (!drag || drag.pointerId !== event.pointerId) {
    return;
  }
  const dx = event.clientX - drag.startX;
  const dy = event.clientY - drag.startY;
  if (Math.abs(dx) + Math.abs(dy) > 4) {
    drag.moved = true;
  }
  if (!drag.moved) {
    return;
  }
  tissueCanvas.classList.add("dragging");
  const currentVector = arcballVector(event);
  const deltaRotation = quatFromVectors(drag.startVector, currentVector);
  state.pickerCamera.rotation = multiplyQuaternions(deltaRotation, drag.startRotation);
  requestPickerRender();
}

function pickerPointerUp(event) {
  const drag = state.pickerDrag;
  if (!drag || drag.pointerId !== event.pointerId) {
    return;
  }
  state.pickerDrag = null;
  tissueCanvas.classList.remove("dragging");
  try {
    tissueCanvas.releasePointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture can already be released by the browser.
  }
  if (!drag.moved) {
    selectPickerPoint(event);
  }
}

function pickerPointerCancel(event) {
  if (state.pickerDrag?.pointerId === event.pointerId) {
    state.pickerDrag = null;
    tissueCanvas.classList.remove("dragging");
  }
}

inspectBtn.addEventListener("click", inspect);
simulateBtn.addEventListener("click", simulate);
pickModelBtn.addEventListener("click", () => modelFileInput.click());
pickTissueBtn.addEventListener("click", () => tissueFileInput.click());
browseModelBtn.addEventListener("click", () => openBrowser("model"));
browseTissueBtn.addEventListener("click", () => openBrowser("tissue"));
closeModalBtn.addEventListener("click", () => {
  browserModal.hidden = true;
});
goPathBtn.addEventListener("click", () => {
  loadBrowserPath(state.browsingKind, modalPathInput.value).catch((error) => setStatus(error.message));
});
modalPathInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadBrowserPath(state.browsingKind, modalPathInput.value).catch((error) => setStatus(error.message));
  }
});

modelFileInput.addEventListener("change", () => {
  const file = modelFileInput.files?.[0];
  if (file) {
    uploadFile("model", file).catch((error) => setStatus(error.message));
  }
  modelFileInput.value = "";
});

tissueFileInput.addEventListener("change", () => {
  const file = tissueFileInput.files?.[0];
  if (file) {
    uploadFile("tissue", file).catch((error) => setStatus(error.message));
  }
  tissueFileInput.value = "";
});

Object.entries(sourceSliders).forEach(([key, slider]) => {
  bindLinkedNumberAndSlider(fields[key], slider, (value) => value.toFixed(1), () => {
    updatePickerSourceText();
    requestPickerRender();
  });
});

Object.entries(dirSliders).forEach(([key, slider]) => {
  bindLinkedNumberAndSlider(fields[key], slider, (value) => value.toFixed(2), requestPickerRender);
});

document.querySelectorAll("[data-dir]").forEach((button) => {
  button.addEventListener("click", () => {
    const [x, y, z] = button.dataset.dir.split(",").map(Number);
    setDirectionValues([x, y, z]);
  });
});

resetViewBtn.addEventListener("click", resetPickerView);
aimCenterBtn.addEventListener("click", aimDirectionAtCenter);
tissueCanvas.addEventListener("pointerdown", pickerPointerDown);
tissueCanvas.addEventListener("pointermove", pickerPointerMove);
tissueCanvas.addEventListener("pointerup", pickerPointerUp);
tissueCanvas.addEventListener("pointercancel", pickerPointerCancel);
tissueCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = event.deltaY > 0 ? 0.92 : 1.08;
  state.pickerCamera.zoom = clamp(state.pickerCamera.zoom * factor, 0.65, 2.8);
  requestPickerRender();
});
window.addEventListener("resize", debounce(requestPickerRender, 100));

setupDropTarget(fields.modelPath, "model");
setupDropTarget(fields.tissuePath, "tissue");

Object.entries(views).forEach(([axis, view]) => {
  view.slider.addEventListener("input", () => {
    view.label.textContent = `${axis.toUpperCase()} ${view.slider.value}`;
    debouncedSliceLoad(axis);
  });
});

(async function init() {
  try {
    await loadDefaults();
    let restoredPlan = null;
    const planId = localStorage.getItem("mcvm.currentPlanId");
    if (planId) {
      try {
        const response = await fetch(`/api/plans/${encodeURIComponent(planId)}`);
        if (response.ok) {
          const record = await response.json();
          if (record.status === "completed" && record.bestCandidate) {
            restoredPlan = record;
            fields.modelPath.value = record.config.modelPath;
            fields.tissuePath.value = record.config.tissuePath;
            fields.numChannels.value = record.config.numChannels;
            setSourceValues(record.bestCandidate.source);
            setDirectionValues(record.bestCandidate.direction);
            saveSharedSettings();
          }
        }
      } catch (_error) {
        restoredPlan = null;
      }
    }
    await inspect();
    if (restoredPlan) {
      await simulate();
      ["x", "y", "z"].forEach((axis, index) => {
        views[axis].slider.value = Math.round(Number(restoredPlan.config.roi.center[index]));
      });
      await loadAllSlices();
      setStatus(`Plan ${restoredPlan.planId} · ${restoredPlan.bestCandidate.id} restored for realtime simulation.`);
    }
  } catch (error) {
    setStatus(error.message);
  }
})();
