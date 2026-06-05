/** Re-export palette helpers; colors come from bootstrap.palette at runtime. */

export function biomeColor(palette, biomeId) {
  const b = palette.biomes[biomeId] || palette.biomes[0];
  return `rgb(${b[0]},${b[1]},${b[2]})`;
}

export function cropColor(palette, stage, health) {
  if (stage <= 0) return null;
  const h = 0.35 + 0.65 * Math.max(0, Math.min(1, health));
  if (stage >= 1) {
    const c = palette.cropMature;
    return `rgb(${Math.round(c[0] * h)},${Math.round(c[1] * h)},${Math.round(c[2] * h)})`;
  }
  const t = Math.max(0, Math.min(1, stage));
  const lo = palette.cropGrowingLo;
  const hi = palette.cropGrowingHi;
  const r = lo[0] * (1 - t) + hi[0] * t;
  const g = lo[1] * (1 - t) + hi[1] * t;
  const b = lo[2] * (1 - t) + hi[2] * t;
  return `rgb(${Math.round(r * h)},${Math.round(g * h)},${Math.round(b * h)})`;
}

export function structureColor(palette, type) {
  const key = String(type);
  const c = palette.structures[key];
  if (!c) return null;
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function agentColor(palette, isPredator) {
  const c = isPredator ? palette.predator : palette.agent;
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
