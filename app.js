import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const fileEl = document.getElementById("file");
const scanBtn = document.getElementById("scan");
const statusEl = document.getElementById("status");
const powerEl = document.getElementById("power");
const detailEl = document.getElementById("detail");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let faceLandmarker = null;
let currentImageBitmap = null;

async function init() {
  status("モデル読み込み中…");

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    numFaces: 1,
  });

  status("準備OK。画像を選んでください。");
}
init().catch((e) => status("初期化エラー: " + e.message));

fileEl.addEventListener("change", async () => {
  powerEl.textContent = "---";
  detailEl.textContent = "";
  scanBtn.disabled = true;

  const file = fileEl.files?.[0];
  if (!file) return;

  status("画像読み込み中…");

  currentImageBitmap = await createImageBitmap(file);
  drawBitmap(currentImageBitmap);
  scanBtn.disabled = false;

  status("測定ボタンを押してください。");
});

scanBtn.addEventListener("click", async () => {
  if (!faceLandmarker || !currentImageBitmap) return;

  status("解析中…");

  const result = faceLandmarker.detect(currentImageBitmap);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    status("顔が検出できませんでした。");
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
  const power = toBattlePower(metrics);

  powerEl.textContent = power;

  detailEl.textContent =
    `対称性誤差: ${metrics.symErr.toFixed(3)} / ` +
    `縦横比誤差: ${metrics.ratioErr.toFixed(3)} / ` +
    `中心ズレ: ${metrics.centerErr.toFixed(3)}`;

  status("完了");
});

function status(t) {
  statusEl.textContent = t;
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
  const GOLD = 1.618;
  const ratioErr = Math.abs(ratio - GOLD);

  const midX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
  const symEye = Math.abs((midX - leftEyeOuter.x) - (rightEyeOuter.x - midX));
  const symMouth = Math.abs((midX - mouthLeft.x) - (mouthRight.x - midX));
  const symErr = symEye + symMouth;

  const centerErr = Math.abs(noseTip.x - midX);

  return { ratioErr, symErr, centerErr };
}

function toBattlePower(m) {
  const err =
    m.ratioErr * 1.6 +
    m.symErr * 2.2 +
    m.centerErr * 1.8;

  const raw = Math.exp(-3.2 * err);
  const compressed = raw / (raw + 0.22);

  const jitter = (Math.random() - 0.5) * 0.05;
  const final = Math.max(0, Math.min(1, compressed + jitter));

  return Math.round(final * 999);
}
