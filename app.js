import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const fileEl = document.getElementById("file");
const scanBtn = document.getElementById("scan");
const statusEl = document.getElementById("status");
const powerEl = document.getElementById("power");
const detailEl = document.getElementById("detail");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const meterFill = document.getElementById("meterFill");
const meterNeedle = document.getElementById("meterNeedle");
const meterValue = document.getElementById("meterValue");

// 追加：ランク表示があれば使う（無くても動く）
const rankEl = document.getElementById("rank");

let faceLandmarker = null;
let currentImageBitmap = null;

function status(t) {
  statusEl.textContent = t;
}

function setMeter(v01, label) {
  const v = Math.max(0, Math.min(1, v01));
  const pct = Math.round(v * 100);
  meterFill.style.width = pct + "%";

  // 針を left で動かす（親の幅に合わせて%で）
  meterNeedle.style.left = `calc(6px + (100% - 12px) * ${v})`;

  meterValue.textContent = label ?? (pct + "%");
}

/**
 * 0..999（整いスコア）→ 5000..50000+ に拡張
 * - ほとんどは 5000 付近
 * - 上位だけ数万に跳ねる
 */
function bigScoreFrom0to999(p) {
  const x = Math.max(0, Math.min(999, p)) / 999; // 0..1
  const gamma = 5.0; // 大きいほど「普通は5000寄り」「上位だけ跳ねる」
  const big = 5000 + Math.round(Math.pow(x, gamma) * 45000); // 5000..50000
  return big;
}

/**
 * レベル判定（要望通り）
 * ただし下限は「かわいい」に固定する
 */
function rankFromBigScore(big) {
  let r = "イマイチ";
  if (big >= 6500) r = "ふつう";
  if (big >= 9000) r = "かわいい";
  if (big >= 20000) r = "激ヤバ";
  if (big >= 35000) r = "神";

  // 下限を「かわいい」に固定
  if (r === "イマイチ" || r === "ふつう") r = "かわいい";
  return r;
}

async function init() {
  status("モデル読み込み中…");
  setMeter(0.08, "LOADING");

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    numFaces: 1,
  });

  status("準備OK。画像を選んでください。");
  setMeter(0.12, "READY");
}
init().catch((e) => {
  status("初期化エラー: " + e.message);
  setMeter(0.02, "ERROR");
});

fileEl.addEventListener("change", async () => {
  powerEl.textContent = "---";
  if (rankEl) rankEl.textContent = "---";
  detailEl.textContent = "画像を選択してください";
  scanBtn.disabled = true;

  const file = fileEl.files?.[0];
  if (!file) return;

  status("画像読み込み中…");
  setMeter(0.18, "IMAGE");

  currentImageBitmap = await createImageBitmap(file);
  drawBitmap(currentImageBitmap);

  scanBtn.disabled = false;
  status("スキャン開始を押してください。");
  setMeter(0.22, "ARMED");
});

scanBtn.addEventListener("click", async () => {
  if (!faceLandmarker || !currentImageBitmap) return;

  status("解析中…");
  animateScanMeter();

  const result = faceLandmarker.detect(currentImageBitmap);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    status("顔が検出できませんでした。");
    setMeter(0.05, "NO FACE");
    return;
  }

  const lm = result.faceLandmarks[0];
  drawBitmap(currentImageBitmap);

  const du = new DrawingUtils(ctx);
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
  du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS);

  const metrics = computeMetrics(lm);

  // 0..999（整いスコア）
  const p0 = toCutePower(metrics);

  // 5000..50000（表示用スコア）
  const big = bigScoreFrom0to999(p0);
  const rank = rankFromBigScore(big);

  // メーターは 0..999 を使って動かす（見た目が自然）
  const v01 = p0 / 999;

  powerEl.textContent = String(big);
  if (rankEl) rankEl.textContent = rank;

  detailEl.textContent =
    `対称性誤差: ${metrics.symErr.toFixed(3)}\n` +
    `比率誤差(φ=1.618): ${metrics.ratioErr.toFixed(3)}\n` +
    `中心ズレ: ${metrics.centerErr.toFixed(3)}\n` +
    `（※小さいほど幾何学的に整っている）`;

  status("完了");
  setMeter(v01, `${big}pt`);
});

let scanAnim = null;
function animateScanMeter() {
  if (scanAnim) cancelAnimationFrame(scanAnim);
  const start = performance.now();
  const loop = (t) => {
    const p = ((t - start) / 900) % 1;
    const v = 0.25 + 0.60 * (0.5 - 0.5 * Math.cos(p * Math.PI * 2));
    setMeter(v, "SCANNING");
    scanAnim = requestAnimationFrame(loop);
  };
  scanAnim = requestAnimationFrame(loop);

  setTimeout(() => {
    if (scanAnim) cancelAnimationFrame(scanAnim);
    scanAnim = null;
  }, 2000);
}

function drawBitmap(bm) {
  canvas.width = bm.width;
  canvas.height = bm.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bm, 0, 0);
}

function pt(lm, idx) {
  return { x: lm[idx].x, y: lm[idx].y };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function computeMetrics(lm) {
  const leftEyeOuter = pt(lm, 33);
  const rightEyeOuter = pt(lm, 263);
  const noseTip = pt(lm, 1);
  const mouthLeft = pt(lm, 61);
  const mouthRight = pt(lm, 291);
  const chin = pt(lm, 152);
  const leftFace = pt(lm, 234);
  const rightFace = pt(lm, 454);

  const faceW = dist(leftFace, rightFace);
  const faceH = dist(chin, noseTip) * 2.0;

  const ratio = faceH / faceW;
  const PHI = 1.618;
  const ratioErr = Math.abs(ratio - PHI);

  const midX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
  const symEye = Math.abs((midX - leftEyeOuter.x) - (rightEyeOuter.x - midX));
  const symMouth = Math.abs((midX - mouthLeft.x) - (mouthRight.x - midX));
  const symErr = symEye + symMouth;

  const centerErr = Math.abs(noseTip.x - midX);

  return { ratioErr, symErr, centerErr };
}

/**
 * 0..999（整いスコア）
 * - ランダム揺れ(jitter)はほぼ消す（再現性上げる）
 * - でも完全固定だと味気ないなら 0.01 程度まで
 */
function toCutePower(m) {
  const err = m.ratioErr * 1.6 + m.symErr * 2.2 + m.centerErr * 1.8;

  const raw = Math.exp(-3.2 * err);
  const compressed = raw / (raw + 0.22);

  const jitter = (Math.random() - 0.5) * 0.01; // ←小さく
  const final = Math.max(0, Math.min(1, compressed + jitter));

  return Math.round(final * 999);
}
