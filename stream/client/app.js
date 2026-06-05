import {
  biomeColor,
  cropColor,
  structureColor,
  agentColor,
} from "./palette.js";

const canvas = document.getElementById("world");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");
const statusEl = document.getElementById("status");

let bootstrap = null;
let palette = null;
let H = 0;
let W = 0;
let biome = null;
/** @type {Map<number, string>} slot -> color at last position */
const entitySlots = new Map();
/** @type {(number|null)[][]} per-cell overlay color override */
let overlay = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function decodeBiome(b64, h, w) {
  const raw = atob(b64);
  const buf = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
  return buf;
}

function drawBasemap() {
  const scale = Math.min(
    Math.floor(canvas.width / W),
    Math.floor(canvas.height / H),
    4
  );
  canvas.width = W * scale;
  canvas.height = H * scale;
  ctx.imageSmoothingEnabled = false;

  overlay = Array.from({ length: H }, () => Array(W).fill(null));

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const b = biome[y * W + x];
      ctx.fillStyle = biomeColor(palette, b);
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }
}

function cellColor(y, x, layers) {
  if (layers.structureType > 0) {
    const sc = structureColor(palette, layers.structureType);
    if (sc) return sc;
  }
  if (layers.cropStage > 0) {
    const cc = cropColor(palette, layers.cropStage, layers.cropHealth ?? 1);
    if (cc) return cc;
  }
  const b = biome[y * W + x];
  return biomeColor(palette, b);
}

function paintCell(y, x, color) {
  overlay[y][x] = color;
  const scale = canvas.width / W;
  ctx.fillStyle = color;
  ctx.fillRect(x * scale, y * scale, scale, scale);
}

function redrawEntity(slot, y, x, isPredator) {
  const old = entitySlots.get(slot);
  if (old) {
    const [oy, ox] = old.split(",").map(Number);
    paintCell(oy, ox, overlay[oy][ox] || biomeColor(palette, biome[oy * W + ox]));
  }
  const col = agentColor(palette, isPredator);
  entitySlots.set(slot, `${y},${x}`);
  const scale = canvas.width / W;
  ctx.fillStyle = col;
  ctx.fillRect(x * scale, y * scale, scale, scale);
}

function removeEntity(slot) {
  const old = entitySlots.get(slot);
  if (old) {
    const [oy, ox] = old.split(",").map(Number);
    paintCell(oy, ox, overlay[oy][ox] || biomeColor(palette, biome[oy * W + ox]));
  }
  entitySlots.delete(slot);
}

function applyTileDelta(tile) {
  const { y, x, layers } = tile;
  const merged = {
    structureType: layers.structureType ?? 0,
    cropStage: layers.cropStage ?? 0,
    cropHealth: layers.cropHealth ?? 0,
  };
  const color = cellColor(y, x, merged);
  paintCell(y, x, color);
}

function applyStepDelta(msg) {
  if (msg.type !== "StepDelta") return;
  hud.textContent = `t=${msg.t}  living=${msg.nLiving}  season=${msg.seasonPhase?.toFixed(2) ?? "?"}  seq=${msg.seq}`;
  for (const t of msg.tiles || []) applyTileDelta(t);
  for (const e of msg.entities || []) {
    if (e.op === "remove") removeEntity(e.slot);
    else redrawEntity(e.slot, e.y, e.x, e.isPredator);
  }
}

async function postControl(action) {
  await fetch("/api/v1/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
}

async function init() {
  setStatus("Loading bootstrap…");
  const res = await fetch("/api/v1/bootstrap");
  bootstrap = await res.json();
  palette = bootstrap.palette;
  H = bootstrap.H;
  W = bootstrap.W;
  biome = decodeBiome(bootstrap.biome, H, W);
  drawBasemap();
  setStatus("Connecting stream…");

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/v1/stream`);

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "BootstrapRef") return;
    applyStepDelta(msg);
    setStatus(`Live  seq=${msg.seq}`);
  };

  ws.onclose = () => setStatus("Disconnected");
  ws.onerror = () => setStatus("WebSocket error");

  document.getElementById("pause").onclick = () => postControl("pause");
  document.getElementById("play").onclick = () => postControl("play");
  document.getElementById("reset").onclick = () => {
    postControl("reset");
    entitySlots.clear();
    drawBasemap();
  };
}

init().catch((err) => setStatus(String(err)));
