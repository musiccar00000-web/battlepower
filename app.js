import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const fileEl = document.getElementById("file");
const scanBtn = document.getElementById("scan");
const statusEl = document.getElementById("status");
const powerEl = document.getElementById("power");
const detailEl = document.getElementById("detail");
const rankEl = document.getElementById("rank");

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const meterFill = document.getElementById("meterFill");
const meterNeedle = document.getElementById("meterNeedle");
const meterValue = document.getElementById("meterValue");

let faceLandmarker = null;
let currentImageBitmap = null;
let scanAnim = null;

function status(t) {
  statusEl.textContent = t;
}

function setMeter(v01, label) {
  const v = Math.max(0, Math.min(1, v01));
  const pct = Math.round(v * 100);

  meterFill.style.width = pct + "%";
  meterNeedle.style.left = `calc(6px + (100% - 12px) * ${v})`;
  meterValue.textContent = label ?? (pct + "%");
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
  rankEl.textContent = "---";
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

  // ★SCANNING確実停止
  if (scanAnim) cancelAnimationFrame(scanAnim);
  scanAnim = null;

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
  const power = toCutePower(metrics);

  const v01 = Math.min(1, power / 99999);

  powerEl.textContent = power;
  rankEl.textContent = rankFromScore(power);

  detailEl.textContent =
    `対称性誤差: ${metrics.symErr.toFixed(3)}\n` +
    `比率誤差(φ=1.618): ${metrics.ratioErr.toFixed(3)}\n` +
    `中心ズレ: ${metrics.centerErr.toFixed(3)}\n` +
    `（※小さいほど幾何学的に整っている）`;

  status("完了");
  setMeter(v01, `${power}pt`);
});

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

/* =============================
   スコア変換（分布拡張版）
============================= */
function toCutePower(m) {
  const err = m.ratioErr * 1.2 + m.symErr * 2.0 + m.centerErr * 2.6;

  const quality = Math.exp(-1.35 * err);
  const t = Math.pow(quality, 0.55);

  let pts = 5000 + Math.round(t * 45000);

  if (err < 0.075) {
    const bonus = (0.075 - err) * 900000;
    pts = Math.round(pts + bonus);
  }

  pts += Math.round((Math.random() - 0.5) * 120);
  pts = Math.max(5000, Math.min(99999, pts));

  return pts;
}

/* =============================
   判定ラベル
============================= */
function rankFromScore(s){
  if (s >= 70000) return "神";
  if (s >= 45000) return "激ヤバ";
  if (s >= 20000) return "かわいい";
  if (s >= 9000)  return "ふつう";
  return "イマイチ";
}
