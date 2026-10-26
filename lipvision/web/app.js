// LipVision browser client.
// Face tracking runs locally (MediaPipe, GPU/WebGL) for instant overlay graphics;
// only a small lower-face JPEG + 4 landmark points per frame goes to the server.
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
// Flow control: never queue stale frames. 8 unacknowledged frames covers a ~250 ms
// tunnel round trip at 30 fps; bufferedAmount stops a slow uplink from building a backlog.
const MAX_INFLIGHT = 8;
const MAX_BUFFERED = 96 * 1024;
const JPEG_QUALITY = 0.85;

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
const crop = document.createElement("canvas"), cctx = crop.getContext("2d");
const patchCtx = $("patch").getContext("2d");
const spark = $("spark"), sctx = spark.getContext("2d");

const state = {
  cfg: null, ws: null, landmarker: null, inflight: 0, mode: "auto", speaking: false,
  speechEndAt: 0, rtt: 0, sent: 0, lines: new Map(), energy: [], thr: 0.012,
  pttHeld: false, captionTimer: 0, retry: 500,
};

// ------------------------------------------------------------------ helpers
const mean = (mesh, idx) => {
  let x = 0, y = 0;
  for (const i of idx) { x += mesh[i][0]; y += mesh[i][1]; }
  return [x / idx.length, y / idx.length];
};
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function setPill(id, cls, text) {
  const el = $(id);
  el.classList.remove("ok", "warn", "bad");
  if (cls) el.classList.add(cls);
  el.querySelector("em").textContent = text;
}

// --------------------------------------------------------------- websocket
function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  setPill("pill-server", "warn", "connecting…");
  ws.onopen = () => { state.inflight = 0; state.retry = 500; sendCtl({ type: "mode", mode: state.mode }); };
  ws.onclose = () => {
    setPill("pill-server", "bad", "reconnecting…");
    setTimeout(connect, state.retry);
    state.retry = Math.min(state.retry * 2, 8000);
  };
  ws.onmessage = (ev) => onMessage(JSON.parse(ev.data));
}

function sendCtl(msg) {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(msg));
}

function onMessage(msg) {
  switch (msg.type) {
    case "ready": {
      setPill("pill-server", "ok", `${msg.device}${msg.fp16 ? " · fp16" : ""}`);
      const info = { version: msg.version, device: msg.device, checkpoint: msg.checkpoint, "beam size": msg.beam };
      $("sysinfo").innerHTML = Object.entries(info).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      break;
    }
    case "error":
      if (msg.utt) { upsertLine({ ...msg, text: `⚠ ${msg.message}`, ms: 0, dur: 0 }, true); break; }
      setPill("pill-server", "bad", "model error");
      $("empty").textContent = `Server error: ${msg.message}`;
      break;
    case "frame": {
      state.inflight = Math.max(0, state.inflight - 1);
      const rtt = performance.now() - msg.t;
      state.rtt = state.rtt ? state.rtt * 0.9 + rtt * 0.1 : rtt;
      if (state.speaking && !msg.speaking) state.speechEndAt = performance.now();
      state.speaking = msg.speaking;
      $("video-wrap").classList.toggle("speaking", msg.speaking);
      state.thr = msg.thr;
      state.energy.push(msg.energy);
      if (state.energy.length > 150) state.energy.shift();
      if (msg.patch) drawPatch(msg.patch);
      break;
    }
    case "partial": upsertLine(msg, false); break;
    case "final": upsertLine(msg, true); break;
    case "discard": {
      const el = state.lines.get(msg.utt);
      if (el && el.classList.contains("partial")) { el.remove(); state.lines.delete(msg.utt); }
      if ($("caption").classList.contains("partial")) $("caption").classList.remove("show");
      break;
    }
  }
}

// --------------------------------------------------------------- transcript
function upsertLine(msg, final) {
  const box = $("transcript");
  $("empty").hidden = true;
  let el = state.lines.get(msg.utt);
  if (!el) {
    el = document.createElement("div");
    el.className = "line";
    el.innerHTML = `<div class="text"></div><div class="meta"></div>`;
    box.appendChild(el);
    state.lines.set(msg.utt, el);
  }
  const text = msg.text || (final ? "[unclear — try again, facing the camera]" : "…");
  el.querySelector(".text").textContent = text;
  el.classList.toggle("partial", !final);
  if (final) {
    const e2e = state.speechEndAt ? Math.round(performance.now() - state.speechEndAt) : null;
    const when = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    el.querySelector(".meta").textContent =
      `${when} · ${msg.dur}s clip · decode ${msg.ms} ms${e2e !== null && e2e < 10000 ? ` · end-to-end ${e2e} ms` : ""}`;
    if (msg.text && $("tts").checked && "speechSynthesis" in window) {
      speechSynthesis.speak(new SpeechSynthesisUtterance(msg.text.toLowerCase()));
    }
  } else {
    el.querySelector(".meta").textContent = "listening to your lips…";
  }
  showCaption(text, final);
  box.scrollTop = box.scrollHeight;
}

function showCaption(text, final) {
  const c = $("caption");
  c.textContent = text;
  c.classList.add("show");
  c.classList.toggle("partial", !final);
  clearTimeout(state.captionTimer);
  if (final) state.captionTimer = setTimeout(() => c.classList.remove("show"), 4500);
}

function drawPatch(b64) {
  const img = new Image();
  img.onload = () => patchCtx.drawImage(img, 0, 0, 96, 96);
  img.src = `data:image/jpeg;base64,${b64}`;
}

// ------------------------------------------------------------------ drawing
function drawOverlay(mesh, pts, speaking) {
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  if (!mesh) return;
  const m = state.cfg.mesh;
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (const [x, y] of mesh) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }

  // face corner brackets
  const L = Math.max(14, (x1 - x0) * 0.12);
  octx.strokeStyle = "rgba(129,140,248,.85)";
  octx.lineWidth = 3;
  octx.beginPath();
  for (const [cx, cy, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
    octx.moveTo(cx, cy + dy * L); octx.lineTo(cx, cy); octx.lineTo(cx + dx * L, cy);
  }
  octx.stroke();

  // glowing lip contours
  const color = speaking ? "#f472b6" : "#5eead4";
  octx.save();
  octx.shadowColor = color;
  octx.shadowBlur = speaking ? 18 : 10;
  octx.strokeStyle = color;
  octx.lineWidth = 2.5;
  for (const ring of [m.lipOuter, m.lipInner]) {
    octx.beginPath();
    ring.forEach((i, k) => (k ? octx.lineTo(...mesh[i]) : octx.moveTo(...mesh[i])));
    octx.closePath();
    octx.stroke();
  }
  octx.restore();

  // alignment anchors (eyes, nose, mouth)
  octx.fillStyle = "rgba(255,255,255,.9)";
  for (const p of pts) { octx.beginPath(); octx.arc(p[0], p[1], 3, 0, Math.PI * 2); octx.fill(); }
}

function drawSpark() {
  const dpr = window.devicePixelRatio || 1;
  const w = spark.clientWidth * dpr, h = spark.clientHeight * dpr;
  if (spark.width !== w) { spark.width = w; spark.height = h; }
  sctx.clearRect(0, 0, w, h);
  const vals = state.energy, top = Math.max(state.thr * 4, 0.05);
  const y = (v) => h - Math.min(1, v / top) * (h - 6) - 3;
  sctx.strokeStyle = "rgba(251,191,36,.55)";
  sctx.setLineDash([4 * dpr, 4 * dpr]);
  sctx.beginPath(); sctx.moveTo(0, y(state.thr)); sctx.lineTo(w, y(state.thr)); sctx.stroke();
  sctx.setLineDash([]);
  if (vals.length < 2) return;
  const grad = sctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#818cf8"); grad.addColorStop(1, "#5eead4");
  sctx.strokeStyle = grad;
  sctx.lineWidth = 2 * dpr;
  sctx.beginPath();
  vals.forEach((v, i) => {
    const x = (i / 149) * w;
    i ? sctx.lineTo(x, y(v)) : sctx.moveTo(x, y(v));
  });
  sctx.stroke();
}

// ---------------------------------------------------------------- streaming
async function sendFrame(t, mesh, pts, openness) {
  const ws = state.ws;
  if (ws?.readyState !== WebSocket.OPEN || state.inflight >= MAX_INFLIGHT || ws.bufferedAmount > MAX_BUFFERED) return;
  state.inflight++;
  const enc = new TextEncoder();
  let header, jpeg = null;
  if (pts) {
    const W = video.videoWidth, H = video.videoHeight, C = state.cfg.cropSize;
    const eyeMouth = dist([(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2], pts[3]);
    const scale = Math.max(dist(pts[0], pts[1]), eyeMouth / 1.18);  // robust to head yaw
    const side = Math.max(48, 2.6 * scale);
    const sx = pts[3][0] - side / 2, sy = pts[3][1] - side / 2, k = C / side;
    const ax = Math.max(0, sx), ay = Math.max(0, sy);
    const bx = Math.min(W, sx + side), by = Math.min(H, sy + side);
    cctx.fillStyle = "#000";
    cctx.fillRect(0, 0, C, C);
    if (bx > ax && by > ay) {
      cctx.drawImage(video, ax, ay, bx - ax, by - ay, (ax - sx) * k, (ay - sy) * k, (bx - ax) * k, (by - ay) * k);
    }
    const blob = await new Promise((r) => crop.toBlob(r, "image/jpeg", JPEG_QUALITY));
    jpeg = new Uint8Array(await blob.arrayBuffer());
    header = { t, open: openness, pts: pts.map(([x, y]) => [(x - sx) * k, (y - sy) * k]) };
  } else {
    header = { t, open: null, pts: null };
  }
  const h = enc.encode(JSON.stringify(header));
  const buf = new Uint8Array(4 + h.length + (jpeg ? jpeg.length : 0));
  new DataView(buf.buffer).setUint32(0, h.length, true);
  buf.set(h, 4);
  if (jpeg) buf.set(jpeg, 4 + h.length);
  if (ws.readyState === WebSocket.OPEN) { ws.send(buf); state.sent++; } else state.inflight--;
}

function onVideoFrame() {
  const t = performance.now();
  const res = state.landmarker.detectForVideo(video, t);
  const lm = res.faceLandmarks && res.faceLandmarks[0];
  if (lm) {
    const W = video.videoWidth, H = video.videoHeight, m = state.cfg.mesh;
    const mesh = lm.map((p) => [p.x * W, p.y * H]);
    const pts = [mean(mesh, m.rightEye), mean(mesh, m.leftEye), mean(mesh, m.nose), mean(mesh, m.mouth)];
    const [top, bottom, left, right] = m.lipOpen.map((i) => mesh[i]);
    const openness = dist(top, bottom) / (dist(left, right) + 1e-6);
    sendFrame(t, mesh, pts, openness);  // crop + encode first, graphics after
    drawOverlay(mesh, pts, state.speaking);
    setPill("pill-face", "ok", "tracking");
  } else {
    sendFrame(t, null, null, null);
    drawOverlay(null);
    setPill("pill-face", "warn", "no face");
  }
  drawSpark();
}

function frameLoop() {
  if (video.requestVideoFrameCallback) {
    const tick = () => { onVideoFrame(); video.requestVideoFrameCallback(tick); };
    video.requestVideoFrameCallback(tick);
  } else {
    let last = -1;
    const tick = () => {
      if (video.currentTime !== last) { last = video.currentTime; onVideoFrame(); }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  setInterval(() => {
    $("fps").textContent = `${state.sent} fps`;
    state.sent = 0;
    if (state.rtt) setPill("pill-latency", null, `${Math.round(state.rtt)} ms rtt`);
  }, 1000);
}

// -------------------------------------------------------------------- start
async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: FACE_MODEL, delegate },
    runningMode: "VIDEO", numFaces: 1,
    minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch (e) { console.warn("GPU delegate unavailable, using CPU", e); return FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
}

async function start() {
  const btn = $("btn-start");
  btn.disabled = true;
  btn.lastChild.textContent = " Loading face tracker…";
  try {
    const [landmarker, stream] = await Promise.all([
      createLandmarker(),
      navigator.mediaDevices.getUserMedia({
        audio: false,  // silent lip reading: the microphone is never requested
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: "user" },
      }),
    ]);
    state.landmarker = landmarker;
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    crop.width = crop.height = state.cfg.cropSize;
    $("start-screen").classList.add("hidden");
    frameLoop();
  } catch (e) {
    console.error(e);
    btn.disabled = false;
    btn.lastChild.textContent = " Retry";
    $("start-screen").querySelector("p").textContent =
      `Could not start: ${e.message}. Allow camera access (the page must be opened over https).`;
  }
}

// ----------------------------------------------------------------- controls
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  $("btn-ptt").hidden = mode !== "manual";
  sendCtl({ type: "mode", mode });
}

function ptt(down) {
  if (state.mode !== "manual" || down === state.pttHeld) return;
  state.pttHeld = down;
  $("btn-ptt").classList.toggle("held", down);
  if (!down) state.speechEndAt = performance.now();
  sendCtl({ type: down ? "start" : "stop", t: performance.now() });
}

function transcriptText() {
  return [...state.lines.values()].filter((el) => !el.classList.contains("partial"))
    .map((el) => el.querySelector(".text").textContent).join("\n");
}

document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("btn-start").addEventListener("click", start);
$("btn-ptt").addEventListener("pointerdown", () => ptt(true));
["pointerup", "pointerleave", "pointercancel"].forEach((ev) => $("btn-ptt").addEventListener(ev, () => ptt(false)));
window.addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat && state.mode === "manual") { e.preventDefault(); ptt(true); } });
window.addEventListener("keyup", (e) => { if (e.code === "Space") ptt(false); });
$("btn-copy").addEventListener("click", () => navigator.clipboard.writeText(transcriptText()));
$("btn-clear").addEventListener("click", () => {
  state.lines.forEach((el) => el.remove());
  state.lines.clear();
  $("empty").hidden = false;
});
$("btn-save").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([transcriptText()], { type: "text/plain" }));
  a.download = `lipvision-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`;
  a.click();
});

state.cfg = await (await fetch("/api/config")).json();
connect();
