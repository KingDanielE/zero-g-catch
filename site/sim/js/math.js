// Small linear algebra for the controller: 3-vectors, row-major 3x3 matrices (Float64Array(9)),
// quaternions [w, x, y, z], and the dense solves the damped pseudo-inverse needs.

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const deg = (r) => (r * 180) / Math.PI;
export const rad = (d) => (d * Math.PI) / 180;
export const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

// --- 3x3 matrices, row-major --------------------------------------------------------------

export const mat = (R) => Float64Array.from(R);
export const col = (R, j) => [R[j], R[3 + j], R[6 + j]];
export const fromCols = (x, y, z) => Float64Array.of(x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]);
export const matVec = (R, v) => [
  R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
  R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
  R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
];
export const matTVec = (R, v) => [
  R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
  R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
  R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
];
export function matMul(A, B) {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return C;
}
export const transpose = (R) => Float64Array.of(R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]);

// --- quaternions [w, x, y, z] ------------------------------------------------------------

export function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}
export const quatConj = (q) => [q[0], -q[1], -q[2], -q[3]];
export function quatNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
export function axisAngleQuat(axis, angle) {
  const n = norm(axis);
  if (n < 1e-12 || Math.abs(angle) < 1e-15) return [1, 0, 0, 0];
  const s = Math.sin(angle / 2) / n;
  return [Math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s];
}
export function quatToMat(q) {
  const [w, x, y, z] = q;
  return Float64Array.of(
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  );
}
// Same branch structure as MuJoCo's mju_mat2Quat.
export function matToQuat(R) {
  let q;
  const tr = R[0] + R[4] + R[8];
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s];
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    q = [(R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s];
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    q = [(R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s];
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    q = [(R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s];
  }
  return quatNormalize(q);
}
export function quatToRotvec(q) {
  if (q[0] < 0) q = [-q[0], -q[1], -q[2], -q[3]];
  const angle = 2 * Math.acos(Math.min(1, q[0]));
  if (angle <= 1e-9) return [0, 0, 0];
  const s = Math.sqrt(Math.max(1e-12, 1 - q[0] * q[0]));
  return [(q[1] / s) * angle, (q[2] / s) * angle, (q[3] / s) * angle];
}
export const rotvecFromMatrix = (R) => quatToRotvec(matToQuat(R));
export function rotvecToQuat(rv) {
  const a = norm(rv);
  return a < 1e-12 ? [1, 0, 0, 0] : axisAngleQuat(scale(rv, 1 / a), a);
}
// Apply a world-frame rotation vector to orientation q (or rotation matrix R).
export const rotateQuat = (q, rv) => (norm(rv) < 1e-12 ? q.slice() : quatNormalize(quatMul(rotvecToQuat(rv), q)));
export const rotateMat = (R, rv) => (norm(rv) < 1e-12 ? R : matMul(quatToMat(rotvecToQuat(rv)), R));

// --- dense matrices for the 6x7 Jacobian ------------------------------------------------

// Damped pseudo-inverse J^T (J J^T + lambda^2 I)^-1 for a 6 x n row-major J; returns n x 6.
export function dampedPinv(J, n, lambda) {
  const A = new Float64Array(36);
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      let s = i === j ? lambda * lambda : 0;
      for (let k = 0; k < n; k++) s += J[i * n + k] * J[j * n + k];
      A[6 * i + j] = s;
    }
  const Ai = inverse(A, 6);
  const P = new Float64Array(n * 6);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < 6; j++) {
      let s = 0;
      for (let k = 0; k < 6; k++) s += J[k * n + i] * Ai[6 * k + j];
      P[6 * i + j] = s;
    }
  return P;
}

function inverse(A, n) {
  const M = Float64Array.from(A);
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
    if (p !== c)
      for (let k = 0; k < n; k++) {
        [M[c * n + k], M[p * n + k]] = [M[p * n + k], M[c * n + k]];
        [I[c * n + k], I[p * n + k]] = [I[p * n + k], I[c * n + k]];
      }
    const d = M[c * n + c];
    for (let k = 0; k < n; k++) {
      M[c * n + k] /= d;
      I[c * n + k] /= d;
    }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r * n + c];
      if (f === 0) continue;
      for (let k = 0; k < n; k++) {
        M[r * n + k] -= f * M[c * n + k];
        I[r * n + k] -= f * I[c * n + k];
      }
    }
  }
  return I;
}

// y = P x for P (rows x cols).
export function mulVec(P, rows, cols, x) {
  const y = new Array(rows).fill(0);
  for (let i = 0; i < rows; i++) {
    let s = 0;
    for (let j = 0; j < cols; j++) s += P[i * cols + j] * x[j];
    y[i] = s;
  }
  return y;
}

// Null-space projection (I - P J) v for P (n x 6) and J (6 x n).
export function nullProject(P, J, n, v) {
  const Jv = mulVec(J, 6, n, v);
  const PJv = mulVec(P, n, 6, Jv);
  return v.map((x, i) => x - PJv[i]);
}

// --- seeded randomness ------------------------------------------------------------------

// Small, fast, seedable PRNG (mulberry32) with Gaussian samples (Box-Muller).
export class Rng {
  constructor(seed = 1) {
    this.s = seed >>> 0 || 1;
    this.spare = null;
  }
  random() {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(lo, hi) {
    return lo + (hi - lo) * this.random();
  }
  normal(mu = 0, sd = 1) {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return mu + sd * z;
    }
    let u = 0;
    while (u === 0) u = this.random();
    const v = this.random();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return mu + sd * r * Math.cos(2 * Math.PI * v);
  }
}
