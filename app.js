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

let faceLandmarker = null;
let currentImageBitmap = null;

function status(t) {
  statusEl.textContent = t;
}

function setMeter(v01, label) {
  const v = Math.max(0, Math.min(1, v01));
  const pct = Math.round(v * 100);
  meterFill.style.width = pct + "%";

  // 針はバー幅に合わせて移動（6px〜(100%-6px)）
  const bar = meterFill.parentElement.getBoundingClientRect();
  const x = 6 + (bar.width - 12) * v;
  meterNeedle.style.transform = `translateX(${x}px)`;

  meterValue.textContent = label ?? (pct + "%");
}

async function init() {
  status("モデル読み込み中…");
  setMeter(0.08, "LOADING");

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  // iPhoneで安定させたいならGPU→CPUに変える（重いが堅い）
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
  const power = toCutePower(metrics);      // 0..999
  const v01 = power / 999;

  powerEl.textContent = power;

  detailEl.textContent =
    `対称性誤差: ${metrics.symErr.toFixed(3)}\n` +
    `比率誤差(φ=1.618): ${metrics.ratioErr.toFixed(3)}\n` +
    `中心ズレ: ${metrics.centerErr.toFixed(3)}\n` +
    `（※小さいほど幾何学的に整っている）`;

  status("完了");
  setMeter(v01, `${power}pt`);
});

let scanAnim = null;
function animateScanMeter() {
  if (scanAnim) cancelAnimationFrame(scanAnim);
  const start = performance.now();
  const loop = (t) => {
    const p = ((t - start) / 900) % 1;
    // 解析中は0.25〜0.85を往復する
    const v = 0.25 + 0.60 * (0.5 - 0.5 * Math.cos(p * Math.PI * 2));
    setMeter(v, "SCANNING");
    scanAnim = requestAnimationFrame(loop);
  };
  scanAnim = requestAnimationFrame(loop);

  // 2秒で自動停止（結果で上書き）
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

// “事実ベース”：顔ランドマークから幾何学指標を計算
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

// 上位出にくい圧縮：999点が連発しない
function toCutePower(m) {
  const err = m.ratioErr * 1.6 + m.symErr * 2.2 + m.centerErr * 1.8;

  // 誤差→スコア変換（小さいほど高い）
  const raw = Math.exp(-3.2 * err);

  // 上位圧縮（ここが“高すぎが出ない”）
  const compressed = raw / (raw + 0.22);

  // ゆらぎ（演出、ただし小さめ）
  const jitter = (Math.random() - 0.5) * 0.05;
  const final = Math.max(0, Math.min(1, compressed + jitter));

  return Math.round(final * 999);
}
