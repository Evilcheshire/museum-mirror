// Main entry — wires up camera, MediaPipe tracking, Three.js scene, UI menu,
// calibration panel, skeleton overlay, and the photo capture button.

import { Tracker, SEG_CLASS } from "./tracking.js";
import { Scene } from "./scene.js";
import {
  Attachment,
  computeHead,
  computeHelmet,
  computeTorso,
  computeHand,
  LAYER,
} from "./attachments.js";
import { takePhoto } from "./photo.js";

const video           = document.getElementById("camera");
const threeCanvas     = document.getElementById("overlay");
const compositeCanvas = document.getElementById("composite");
const skelCanvas      = document.getElementById("skeleton");
const statusEl        = document.getElementById("status");
const fpsEl           = document.getElementById("fps");

// Realistic try-on compositor — set up at startup, used when toggled on.
let compositeCtx = null;        // 2D ctx of the visible compositeCanvas
let skinCanvas, skinCtx;        // full-res scratch (skin overlay)
let regionCanvas, regionCtx;    // full-res scratch (hand-vicinity region)
let tryonEnabled = false;

// Hair+skin alpha mask at the segmenter's native resolution. Rebuilt ONLY when
// a new segmentation frame arrives (tracked via builtMaskRef), not every frame.
let skinMaskCanvas, skinMaskCtx;
let builtMaskRef = null;
let segTick = 0;          // throttle counter for the segmenter
const SEG_EVERY = 2;      // run segmentation every Nth frame (mask changes slowly)

// Mobile detection — used to warn that the try-on pipeline is heavy on phones,
// and to hide the back-camera toggle on desktops (single front webcam).
const IS_MOBILE = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

let scene;
let tracker;
let catalog = { hats: [], tops: [], exhibits: [] };

const attachments = new Map();
const selected = { hat: null, top: null, rightHand: null, leftHand: null };

// Calibration state
let calibrationMode = false;
let calibSlot = "hat";

// Camera facing mode — "user" (front, mirrored) or "environment" (back)
let currentFacing = "user";

// ---------- Gesture state ----------
// Pinch-based editing in calibration mode. One-hand pinch drags offset X/Y,
// two-hand pinch scales (distance) and rotates Z (angle of line between them).
// Pinch detection thresholds. Tightened for control: a clear pinch is required,
// and the hand must be fully in frame and held for a few frames before a gesture
// engages — so a hand that's just relaxed, partly off-camera, or passing by
// doesn't trigger anything.
const GESTURE_PINCH_GRAB    = 0.45;  // tip-distance / hand-size to ENTER pinch
const GESTURE_PINCH_RELEASE = 0.62;  // hysteresis to EXIT pinch (avoids flicker)
const GESTURE_ENGAGE_FRAMES = 4;     // consecutive pinched frames before acting
const GESTURE_EDGE_MARGIN   = 0.04;  // reject hands whose key points leave frame
const GESTURE_TRANSLATE_SENS = 1.6;  // 1 full screen swipe ≈ 1.6 model units
const GESTURE_ROTATE_SENS = 360;     // 1 full screen swipe ≈ 360° (exhibits)
const GESTURE_SCALE_MIN = 0.1;
// Two-hand scale needs a non-trivial starting separation, otherwise tiny
// movements between near-overlapping pinches multiply into huge scale jumps.
const GESTURE_TWO_HAND_MIN_DIST = 0.08; // 8% of image width

const gestureState = {
  mode: "idle",              // "idle" | "translate" | "rotate-xy" | "scale-rotate"
  pinchedHands: new Set(),   // hand indices pinched THIS frame (for hysteresis)
  pinchStreak: new Map(),    // hand index → consecutive pinched frames (engage delay)
  startPinches: null,        // initial pinch positions for the active gesture
  startDistance: 0,
  startAngle: 0,
  startConfig: null,         // snapshot of att.config at gesture start
};

function resetGesture() {
  gestureState.mode = "idle";
  gestureState.pinchedHands.clear();
  gestureState.pinchStreak.clear();
  gestureState.startPinches = null;
  gestureState.startConfig = null;
}

// Stored per-frame data for skeleton overlay
let lastPose = null;
let lastHands = [];

let frames = 0;
let lastFpsT = 0;
let skelCtx;

// ---------- Startup ----------

async function init() {
  // Apply UI-hidden state immediately so the loading screen matches.
  try {
    if (localStorage.getItem("museum-mirror-ui-hidden") === "1") {
      document.body.classList.add("ui-hidden");
    }
  } catch {}

  // Wire the shell controls (panel hide, photo button) BEFORE the camera so
  // they work even on machines without a webcam — otherwise an early failure
  // in the camera step left every button unbound.
  bindShellUI();

  let stage = "ініціалізації";
  try {
    // ---- catalog ----
    stage = "завантаження каталогу";
    statusEl.textContent = "Завантаження каталогу...";
    catalog = await fetch("items.json").then((r) => {
      if (!r.ok) throw new Error("items.json не знайдено");
      return r.json();
    });

    // ---- camera ----
    stage = "доступу до камери";
    statusEl.textContent = "Запит доступу до камери...";
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Браузер не підтримує getUserMedia (потрібен HTTPS і сучасний браузер)");
    }
    const stream = await openCamera();
    video.srcObject = stream;
    await new Promise((r) => (video.onloadedmetadata = r));
    await video.play();

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) throw new Error("Камера повернула порожній кадр");

    threeCanvas.width     = w;
    threeCanvas.height    = h;
    compositeCanvas.width = w;
    compositeCanvas.height= h;
    skelCanvas.width      = w;
    skelCanvas.height     = h;
    skelCtx      = skelCanvas.getContext("2d");
    compositeCtx = compositeCanvas.getContext("2d");

    // Offscreen canvases for the realistic try-on compositor.
    skinCanvas = document.createElement("canvas");
    skinCanvas.width = w; skinCanvas.height = h;
    skinCtx = skinCanvas.getContext("2d");
    regionCanvas = document.createElement("canvas");
    regionCanvas.width = w; regionCanvas.height = h;
    regionCtx = regionCanvas.getContext("2d");
    skinMaskCanvas = document.createElement("canvas"); // native seg resolution
    skinMaskCtx    = skinMaskCanvas.getContext("2d");

    scene = new Scene(threeCanvas, w, h);

    // ---- mediapipe ----
    stage = "завантаження MediaPipe";
    statusEl.textContent = "Завантаження моделей MediaPipe...";
    tracker = new Tracker();
    await tracker.init();

    // ---- 3d models ----
    stage = "завантаження 3D-моделей";
    statusEl.textContent = "Завантаження 3D-моделей...";
    await loadCatalog();

    buildMenu();
    bindUI();

    statusEl.textContent = "";
    requestAnimationFrame(loop);
  } catch (err) {
    const name = err?.name ? `[${err.name}] ` : "";
    statusEl.textContent = `Помилка на етапі ${stage}: ${name}${err?.message || err}`;
    console.error(`[init/${stage}]`, err);
  }
}

// Camera open with progressive fallback. iOS Safari (and some Android browsers)
// throw NotFoundError / OverconstrainedError when the requested resolution
// isn't available. Try high-res first, then medium, then any camera.
async function openCamera(facing = currentFacing) {
  const attempts = [
    { video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: { facingMode: { ideal: facing }, width: { ideal: 640 },  height: { ideal: 480 } }, audio: false },
    { video: { facingMode: facing }, audio: false },
    { video: true, audio: false },
  ];
  let lastErr;
  for (const c of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(c);
    } catch (e) {
      lastErr = e;
      console.warn("getUserMedia спроба не вдалась:", c, e?.name, e?.message);
    }
  }
  throw lastErr ?? new Error("Камера недоступна");
}

// Switch between front and back cameras at runtime.
// Steps: stop the old stream's tracks, ask getUserMedia for the new facing,
// flip the scene.mirror flag and the CSS mirror on the <video> element so
// imageToWorld + the head/torso bases stay consistent with what's on screen.
async function switchCamera() {
  const btn = document.getElementById("camera-toggle");
  const newFacing = currentFacing === "user" ? "environment" : "user";
  btn.disabled = true;
  const prevText = btn.textContent;
  btn.textContent = "Перемикання...";
  try {
    if (video.srcObject) {
      for (const t of video.srcObject.getTracks()) t.stop();
    }
    const stream = await openCamera(newFacing);
    video.srcObject = stream;
    // addEventListener + {once} is more reliable than `onloadedmetadata = r`,
    // which can miss the event if it fires before the assignment.
    await new Promise((resolve) => {
      video.addEventListener("loadedmetadata", resolve, { once: true });
    });
    await video.play();

    // Different cameras frequently report different resolutions (front vs back,
    // landscape vs portrait). Sync everything to the new size or the 3D camera
    // aspect goes wrong and MediaPipe gets a mismatched buffer.
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w && h && (threeCanvas.width !== w || threeCanvas.height !== h)) {
      threeCanvas.width = w;
      threeCanvas.height = h;
      skelCanvas.width  = w;
      skelCanvas.height = h;
      scene.setSize(w, h);
    }

    currentFacing = newFacing;
    const mirror = newFacing === "user";
    scene.mirror = mirror;
    video.classList.toggle("no-mirror", !mirror);
    btn.textContent = mirror ? "Камера: фронтальна" : "Камера: задня";
    builtMaskRef = null; // segmentation mask is stale after a camera swap
  } catch (e) {
    console.error("Не вдалось перемкнути камеру:", e);
    statusEl.textContent = `Помилка камери: ${e.message}`;
    setTimeout(() => (statusEl.textContent = ""), 3000);
    btn.textContent = prevText;
  } finally {
    btn.disabled = false;
  }
}

// ---------- Photo capture (with shutter flash + button state) ----------

async function handlePhotoClick() {
  const btn = document.getElementById("photo-btn");
  const flash = document.getElementById("photo-flash");
  if (!btn || btn.disabled) return;

  const originalText = btn.textContent;

  // Trigger the white-flash overlay (CSS animation handles the timing)
  if (flash) {
    flash.classList.remove("firing");
    // force reflow so removing+adding the class restarts the animation
    void flash.offsetWidth;
    flash.classList.add("firing");
  }

  // Lock the button and show the "saving" shimmer
  btn.disabled = true;
  btn.classList.add("saving");
  btn.textContent = "Збереження…";

  try {
    // In try-on mode the composite canvas already contains video + 3D blended
    // with skin overlays, so we capture it directly (composedOnly=true).
    if (tryonEnabled) {
      await takePhoto(video, compositeCanvas, false, true);
    } else {
      await takePhoto(video, threeCanvas, scene?.mirror !== false, false);
    }
    // Brief success state
    btn.classList.remove("saving");
    btn.classList.add("saved");
    btn.textContent = "Збережено ✓";
    await new Promise((r) => setTimeout(r, 900));
  } catch (e) {
    btn.classList.remove("saving");
    btn.textContent = `Помилка: ${e.message}`;
    await new Promise((r) => setTimeout(r, 1600));
  } finally {
    btn.classList.remove("saving", "saved");
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

// ---------- Realistic try-on compositor (full 3-layer pipeline) ----------
//
// Combines three techniques every frame:
//   1) CLOTHES ERASURE   — segmenter's "clothes" class painted over with the
//                          user's own skin colour BEFORE drawing virtual ones
//   2) SILHOUETTE CLIP   — virtual TOPS clipped to the body outline so the
//                          shirt doesn't bleed into background
//   3) SKIN OVERLAY      — hair/face/body-skin pixels redrawn ON TOP so a
//                          hand passing in front correctly covers the shirt
//
// Heavy: 3 Three.js render passes + 4 per-pixel mask loops. On a mid-range
// phone expect ~5-10 fps. On desktop 25-30 fps. The toggle confirms before
// enabling on mobile so the user is warned.
//
// Layer render order on compositeCanvas (bottom → top):
//   body (video with clothes erased) → TOPS (clipped) → HAND items → skin
//   overlay (hair/face/skin) → HATS / helmets

async function toggleTryon() {
  const btn = document.getElementById("tryon-toggle");
  if (!btn) return;

  if (!tryonEnabled) {
    // Mobile warning — the segmenter + extra render pass lower FPS.
    if (IS_MOBILE) {
      const ok = window.confirm(
        "Оклюзія рук вмикає сегментер MediaPipe.\n\n" +
        "На мобільному це помітно знизить FPS і гріє телефон.\n\n" +
        "Все одно увімкнути?"
      );
      if (!ok) return;
    }

    if (!tracker.segmenter) {
      btn.disabled = true;
      btn.textContent = "Завантаження…";
      try { await tracker.ensureSegmenter(); }
      catch (e) {
        console.error("ensureSegmenter:", e);
        btn.textContent = "Примірка: помилка";
        btn.disabled = false;
        return;
      }
      btn.disabled = false;
    }
    tryonEnabled = true;
    builtMaskRef = null; // force a fresh mask build on the next frame
    btn.classList.add("active");
    btn.textContent = "Примірка ●";
    threeCanvas.classList.add("hidden");
    compositeCanvas.classList.remove("hidden");
  } else {
    tryonEnabled = false;
    btn.classList.remove("active");
    btn.textContent = "Примірка ○";
    threeCanvas.classList.remove("hidden");
    compositeCanvas.classList.add("hidden");
  }
}

// Draw <video> onto a 2D ctx, mirrored if the displayed video is mirrored.
function drawMirroredVideo(dstCtx, w, h) {
  dstCtx.clearRect(0, 0, w, h);
  if (scene.mirror) {
    dstCtx.save();
    dstCtx.translate(w, 0);
    dstCtx.scale(-1, 1);
    dstCtx.drawImage(video, 0, 0, w, h);
    dstCtx.restore();
  } else {
    dstCtx.drawImage(video, 0, 0, w, h);
  }
}

// Build an alpha-only mask for the given segmenter classes into `canvas`
// (at native seg resolution). Allocates a fresh ImageData each call — but
// callers only invoke this when a NEW segmentation arrives, not every frame.
function buildMaskCanvas(mask, classIds, canvas, ctx) {
  const sw = mask.width, sh = mask.height;
  if (canvas.width !== sw || canvas.height !== sh) { canvas.width = sw; canvas.height = sh; }
  const img = ctx.createImageData(sw, sh);
  const px = img.data, src = mask.data;
  const single = classIds.length === 1;
  const c0 = classIds[0];
  const set = single ? null : new Set(classIds);
  for (let i = 0, j = 0; i < src.length; i++, j += 4) {
    const hit = single ? src[i] === c0 : set.has(src[i]);
    px[j] = 255; px[j + 1] = 255; px[j + 2] = 255; px[j + 3] = hit ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
}

// Draw a native-res mask canvas onto a full-res ctx, mirrored to match display.
function blitFull(srcCanvas, dstCtx, w, h) {
  if (scene.mirror) {
    dstCtx.save();
    dstCtx.translate(w, 0);
    dstCtx.scale(-1, 1);
    dstCtx.drawImage(srcCanvas, 0, 0, w, h);
    dstCtx.restore();
  } else {
    dstCtx.drawImage(srcCanvas, 0, 0, w, h);
  }
}

// Fill the region canvas with white discs around every detected hand landmark,
// so the skin overlay can be limited to the VICINITY OF THE HANDS. This is the
// key fix for "dress clipped at the straps": a static shoulder/neck must NOT
// occlude the dress, only a hand actually passing in front of it.
function buildHandRegion(w, h) {
  regionCtx.clearRect(0, 0, w, h);
  regionCtx.fillStyle = "#fff";
  for (const hand of lastHands) {
    if (!hand || hand.length < 21) continue;
    const wrist = hand[0], mid = hand[9];
    const hs = Math.hypot((wrist.x - mid.x) * w, (wrist.y - mid.y) * h) || 30;
    const r = hs * 2.4; // a bit larger than the palm to cover fingers/wrist
    for (const lm of hand) {
      const x = (scene.mirror ? (1 - lm.x) : lm.x) * w;
      const y = lm.y * h;
      regionCtx.beginPath();
      regionCtx.arc(x, y, r, 0, Math.PI * 2);
      regionCtx.fill();
    }
  }
}

function renderRealistic(t) {
  const W = compositeCanvas.width;
  const H = compositeCanvas.height;

  compositeCtx.clearRect(0, 0, W, H);
  drawMirroredVideo(compositeCtx, W, H);

  // Hand occlusion only matters when a dress is worn AND a hand is in frame.
  // Otherwise fall through to a single cheap render pass with no segmenter —
  // this is the main optimisation (helmet-only / hands-down ≈ normal-mode FPS).
  const occlude = !!selected.top && lastHands.length > 0;

  if (!occlude) {
    scene.camera.layers.enableAll();
    scene.renderer.clear();
    scene.render();
    compositeCtx.drawImage(threeCanvas, 0, 0, W, H);
    return;
  }

  // Segmentation — throttled to every Nth frame; cached mask reused between.
  let mask = tracker.lastMask;
  if (segTick++ % SEG_EVERY === 0) mask = tracker.segment(video, t);
  if (mask && mask !== builtMaskRef) {
    buildMaskCanvas(mask, [
      SEG_CLASS.HAIR, SEG_CLASS.BODY_SKIN, SEG_CLASS.FACE_SKIN,
    ], skinMaskCanvas, skinMaskCtx);
    builtMaskRef = mask;
  }

  // (1) TOPS (dress) over the body
  scene.camera.layers.set(LAYER.TOPS);
  scene.renderer.clear();
  scene.render();
  compositeCtx.drawImage(threeCanvas, 0, 0, W, H);

  // (2) skin overlay, restricted to the HAND VICINITY — only a hand crossing in
  //     front of the dress re-appears over it (skin ∩ skin-mask ∩ hand-region).
  if (builtMaskRef) {
    buildHandRegion(W, H);
    drawMirroredVideo(skinCtx, W, H);
    skinCtx.save();
    skinCtx.globalCompositeOperation = "destination-in";
    blitFull(skinMaskCanvas, skinCtx, W, H); // keep skin pixels
    skinCtx.drawImage(regionCanvas, 0, 0);   // ∩ hand vicinity
    skinCtx.restore();
    compositeCtx.drawImage(skinCanvas, 0, 0);
  }

  // (3) HATS + HAND items on top (helmet above hair, weapon fully visible)
  scene.camera.layers.set(LAYER.HATS);
  scene.camera.layers.enable(LAYER.HAND_ITEMS);
  scene.renderer.clear();
  scene.render();
  compositeCtx.drawImage(threeCanvas, 0, 0, W, H);
}

// ---------- UI visibility ----------

// Hide / show the controls sidebar. The toggle button stays in the masthead
// so the user can always bring the UI back. Preference is persisted.
function setUIHidden(hidden) {
  document.body.classList.toggle("ui-hidden", hidden);
  try { localStorage.setItem("museum-mirror-ui-hidden", hidden ? "1" : "0"); } catch {}
  const btn = document.getElementById("ui-toggle");
  if (btn) {
    btn.textContent = hidden ? "◑" : "◐";
    btn.title = hidden ? "Показати панель" : "Приховати панель";
    btn.classList.toggle("active", hidden);
  }
}

function initUIToggle() {
  let saved = false;
  try { saved = localStorage.getItem("museum-mirror-ui-hidden") === "1"; } catch {}
  setUIHidden(saved);
  document.getElementById("ui-toggle")?.addEventListener("click", () => {
    setUIHidden(!document.body.classList.contains("ui-hidden"));
  });
}


// ---------- Catalog ----------

async function loadCatalog() {
  // Tag each item with its slot so the compositor can route it to the right
  // render layer (hats above hair, tops with silhouette clip, hand-items
  // below skin so the hand wraps the grip).
  const groups = [
    { items: catalog.hats     || [], slot: "hat"  },
    { items: catalog.tops     || [], slot: "top"  },
    { items: catalog.exhibits || [], slot: "hand" },
  ];
  for (const { items, slot } of groups) {
    for (const cfg of items) {
      try {
        const att = new Attachment(scene, cfg, slot);
        await att.load();
        attachments.set(cfg.id, att);
      } catch (e) {
        console.warn(`Не вдалося завантажити "${cfg.id}" (${cfg.file}):`, e);
      }
    }
  }
}

// ---------- Menu ----------

function buildMenu() {
  const hats    = (catalog.hats    || []).filter((c) => attachments.has(c.id));
  const tops    = (catalog.tops    || []).filter((c) => attachments.has(c.id));
  const exhibits= (catalog.exhibits|| []).filter((c) => attachments.has(c.id));

  renderPanel("hat-panel",        "Головні убори",        hats,    (id) => setSelection("hat", id));
  renderPanel("top-panel",        "Верхній одяг",          tops,    (id) => setSelection("top", id));
  renderPanel("right-hand-panel", "Експонат — права рука", exhibits,(id) => setSelection("rightHand", id));
  renderPanel("left-hand-panel",  "Експонат — ліва рука",  exhibits,(id) => setSelection("leftHand", id));
}

function renderPanel(panelId, title, items, onSelect) {
  const panel = document.getElementById(panelId);
  panel.innerHTML = `<h3>${title}</h3>`;

  if (items.length === 0) {
    const note = document.createElement("div");
    note.className = "empty-note";
    note.textContent = "Додайте моделі та оновіть items.json";
    panel.appendChild(note);
    return;
  }

  const buttons = document.createElement("div");
  buttons.className = "menu-buttons";

  const off = document.createElement("button");
  off.textContent = "Без";
  off.className = "off active";
  off.onclick = () => {
    buttons.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
    off.classList.add("active");
    onSelect(null);
  };
  buttons.appendChild(off);

  items.forEach((item) => {
    const b = document.createElement("button");
    b.textContent = item.label;
    b.onclick = () => {
      buttons.querySelectorAll("button").forEach((bb) => bb.classList.remove("active"));
      b.classList.add("active");
      onSelect(item.id);
    };
    buttons.appendChild(b);
  });

  panel.appendChild(buttons);
}

function setSelection(slot, id) {
  const prevId = selected[slot];
  if (prevId && attachments.has(prevId)) attachments.get(prevId).hide();
  selected[slot] = id;
  if (id && attachments.has(id)) attachments.get(id).show();
  if (calibrationMode) syncCalibSliders();
}

// ---------- UI binding ----------

// Controls that work with or without a camera — bound at startup so a
// missing webcam doesn't leave the whole UI dead.
function bindShellUI() {
  initUIToggle();
  // photo handler self-guards: if the camera never started, takePhoto() shows
  // a clear "Камера не готова" message instead of failing silently.
  document.getElementById("photo-btn").onclick = handlePhotoClick;
}

// Camera/session-dependent controls — bound only after init succeeds.
function bindUI() {
  // Camera switching only makes sense on phones/tablets (a back camera).
  // Desktops have a single front-facing webcam, so hide the toggle there.
  const camBtn = document.getElementById("camera-toggle");
  if (camBtn) {
    if (IS_MOBILE) camBtn.addEventListener("click", switchCamera);
    else camBtn.classList.add("hidden");
  }
  document.getElementById("tryon-toggle")?.addEventListener("click", toggleTryon);

  // Mode toggle
  const modeBtn = document.getElementById("mode-toggle");
  if (modeBtn) {
    modeBtn.onclick = async () => {
      const next = tracker.currentMode === "lite" ? "full" : "lite";
      modeBtn.disabled = true;
      modeBtn.textContent = "Перемикання…";
      await tracker.setMode(next);
      modeBtn.disabled = false;
      modeBtn.textContent = next === "full" ? "Режим: повний ●" : "Режим: легкий ○";
      modeBtn.classList.toggle("active", next === "full");
    };
  }

  // Calibration toggle
  const calibBtn = document.getElementById("calib-toggle");
  const calibPanel = document.getElementById("calib-panel");
  if (calibBtn && calibPanel) {
    calibBtn.onclick = () => {
      calibrationMode = !calibrationMode;
      calibBtn.classList.toggle("active", calibrationMode);
      calibBtn.textContent = calibrationMode ? "Вийти з калібрування" : "Калібрування";
      calibPanel.classList.toggle("hidden", !calibrationMode);
      skelCanvas.classList.toggle("hidden", !calibrationMode);
      if (calibrationMode) syncCalibSliders();
    };
  }

  // Calibration slot selector
  const slotSel = document.getElementById("calib-slot");
  if (slotSel) {
    slotSel.onchange = () => {
      calibSlot = slotSel.value;
      syncCalibSliders();
    };
  }

  // Calibration sliders
  const sliderIds = ["calib-ox","calib-oy","calib-oz","calib-rx","calib-ry","calib-rz","calib-scale"];
  for (const id of sliderIds) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", onCalibInput);
  }

  document.getElementById("calib-reset")?.addEventListener("click", onCalibReset);
}

// ---------- Calibration ----------

function syncCalibSliders() {
  const id  = selected[calibSlot];
  const att = id ? attachments.get(id) : null;
  const noItem = document.getElementById("calib-noitem");
  const sliders = document.getElementById("calib-sliders");

  if (!att) {
    noItem?.classList.remove("hidden");
    sliders?.classList.add("hidden");
    return;
  }
  noItem?.classList.add("hidden");
  sliders?.classList.remove("hidden");

  const cfg = att.config;
  const off = cfg.offset   ?? [0, 0, 0];
  const rot = cfg.rotation ?? [0, 0, 0];
  const sc  = cfg.scale    ?? 1.0;

  applySlider("calib-ox",    off[0]);
  applySlider("calib-oy",    off[1]);
  applySlider("calib-oz",    off[2]);
  applySlider("calib-rx",    rot[0]);
  applySlider("calib-ry",    rot[1]);
  applySlider("calib-rz",    rot[2]);
  applySlider("calib-scale", sc);
}

function applySlider(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = value;
  const disp = document.getElementById(id + "-val");
  if (disp) disp.textContent = Number(value).toFixed(2);
}

function onCalibInput() {
  const id  = selected[calibSlot];
  if (!id) return;
  const att = attachments.get(id);
  if (!att) return;

  const g = (sid) => parseFloat(document.getElementById(sid)?.value ?? 0);
  att.config.offset   = [g("calib-ox"), g("calib-oy"), g("calib-oz")];
  att.config.rotation = [g("calib-rx"), g("calib-ry"), g("calib-rz")];
  att.config.scale    = g("calib-scale");
  att.applyConfig();

  // Update readouts
  ["calib-ox","calib-oy","calib-oz","calib-rx","calib-ry","calib-rz","calib-scale"].forEach((sid) => {
    const el = document.getElementById(sid);
    const disp = document.getElementById(sid + "-val");
    if (el && disp) disp.textContent = Number(el.value).toFixed(2);
  });
}

function onCalibReset() {
  const id  = selected[calibSlot];
  if (!id) return;
  const att = attachments.get(id);
  if (!att) return;
  att.config.offset   = [0, 0, 0];
  att.config.rotation = [0, 0, 0];
  att.config.scale    = 1.0;
  att.applyConfig();
  syncCalibSliders();
}

// ---------- Skeleton overlay ----------

// Key connections between pose landmarks for the skeleton ghost
const POSE_CONN = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [0,7],[0,8],
];

const HAND_CONN = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawSkeleton() {
  skelCtx.clearRect(0, 0, skelCanvas.width, skelCanvas.height);
  if (!calibrationMode) return;

  const W = skelCanvas.width;
  const H = skelCanvas.height;
  // Match the X-mirroring of the displayed video — in selfie mode the video
  // is CSS-flipped, so we flip landmark X here too; in back-camera mode the
  // video is shown natural, so we pass through.
  const mirror = scene?.mirror !== false;
  const px = (lm) => (mirror ? (1 - lm.x) : lm.x) * W;
  const py = (lm) => lm.y * H;

  function drawSet(landmarks, connections, stroke, fill) {
    skelCtx.strokeStyle = stroke;
    skelCtx.fillStyle   = fill;
    skelCtx.lineWidth   = 1.5;
    for (const [a, b] of connections) {
      const la = landmarks[a], lb = landmarks[b];
      if (!la || !lb) continue;
      skelCtx.beginPath();
      skelCtx.moveTo(px(la), py(la));
      skelCtx.lineTo(px(lb), py(lb));
      skelCtx.stroke();
    }
    for (const lm of landmarks) {
      if (!lm) continue;
      skelCtx.beginPath();
      skelCtx.arc(px(lm), py(lm), 3, 0, Math.PI * 2);
      skelCtx.fill();
    }
  }

  if (lastPose) {
    drawSet(lastPose, POSE_CONN,
      "rgba(200,155,63,0.55)", "rgba(200,155,63,0.85)");
  }
  for (const hand of lastHands) {
    drawSet(hand, HAND_CONN,
      "rgba(154,180,255,0.55)", "rgba(154,180,255,0.85)");
  }

  // Pinch indicators — small ring on each hand, filled and bigger while pinched.
  // While a gesture is active, also draw a line from start to current pinch
  // so the user gets a clear "I'm being grabbed" cue.
  for (let i = 0; i < lastHands.length; i++) {
    const p = readHandPinch(lastHands[i]);
    if (!p) continue;
    const isPinched = gestureState.pinchedHands.has(i);
    const cx = (mirror ? (1 - p.x) : p.x) * W;
    const cy = p.y * H;

    skelCtx.beginPath();
    skelCtx.arc(cx, cy, isPinched ? 16 : 10, 0, Math.PI * 2);
    skelCtx.fillStyle = isPinched ? "rgba(200,155,63,0.55)" : "rgba(200,155,63,0.15)";
    skelCtx.fill();
    skelCtx.lineWidth = 1.5;
    skelCtx.strokeStyle = isPinched ? "rgba(200,155,63,0.95)" : "rgba(200,155,63,0.5)";
    skelCtx.stroke();
  }

  // Mode label at the top so the user knows what gesture is being recognised.
  if (gestureState.mode !== "idle") {
    const label =
      gestureState.mode === "translate"  ? "Жест: переміщення" :
      gestureState.mode === "rotate-xy"  ? "Жест: поворот"     :
                                           "Жест: масштаб + поворот";
    skelCtx.font = "16px 'Inter', system-ui, sans-serif";
    skelCtx.fillStyle = "rgba(200,155,63,0.95)";
    skelCtx.textAlign = "center";
    skelCtx.fillText(label, W / 2, 28);
  }
}

// ---------- Gestures ----------

// Detect pinch on a hand. Pinch = thumb tip and index tip close together
// relative to the hand size (so it works regardless of how far the hand is
// from the camera). Returns { x, y, ratio } in normalized image coords.
function readHandPinch(handLm) {
  if (!handLm || handLm.length < 21) return null;
  const thumb = handLm[4];   // THUMB_TIP
  const index = handLm[8];   // INDEX_TIP
  const wrist = handLm[0];   // WRIST
  const middle = handLm[9];  // MIDDLE_MCP
  if (!thumb || !index || !wrist || !middle) return null;

  // Reject hands that are (partly) outside the frame. When a hand leaves the
  // view MediaPipe extrapolates landmarks off-screen, which produced phantom
  // pinches. Require all key points to be well inside the frame.
  const m = GESTURE_EDGE_MARGIN;
  for (const lm of [thumb, index, wrist, middle]) {
    if (lm.x < m || lm.x > 1 - m || lm.y < m || lm.y > 1 - m) return null;
  }

  const tipDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
  const handSize = Math.hypot(wrist.x - middle.x, wrist.y - middle.y) || 1e-6;
  return {
    x: (thumb.x + index.x) / 2,
    y: (thumb.y + index.y) / 2,
    ratio: tipDist / handSize,
  };
}

function isPinchedNow(pinch, wasPinched) {
  if (!pinch) return false;
  const t = wasPinched ? GESTURE_PINCH_RELEASE : GESTURE_PINCH_GRAB;
  return pinch.ratio < t;
}

// Apply pinch gestures to the currently-calibrated item.
// Called every frame from the loop when calibrationMode is on.
function updateGesture(hands) {
  const id = selected[calibSlot];
  const att = id ? attachments.get(id) : null;
  if (!att) { resetGesture(); return; }

  // Build the list of pinches that count THIS frame. A pinch only counts after
  // it's been held for GESTURE_ENGAGE_FRAMES consecutive frames (engage delay),
  // and readHandPinch already rejects hands that are leaving the frame.
  const active = [];
  const nowPinched = new Set();
  const nextStreak = new Map();
  for (let i = 0; i < hands.length; i++) {
    const p = readHandPinch(hands[i]);
    if (!p) continue;
    const was = gestureState.pinchedHands.has(i);
    if (isPinchedNow(p, was)) {
      nowPinched.add(i);
      const streak = (gestureState.pinchStreak.get(i) || 0) + 1;
      nextStreak.set(i, streak);
      if (streak >= GESTURE_ENGAGE_FRAMES) active.push(p); // engaged only after delay
    }
  }
  gestureState.pinchedHands = nowPinched;
  gestureState.pinchStreak = nextStreak;

  if (active.length === 0) {
    gestureState.mode = "idle";
    gestureState.startConfig = null;
    return;
  }

  // Exhibits (hand slots) are locked in the palm — they can't be translated,
  // only rotated. A single pinch-drag rotates them freely (X = pitch, Y = yaw);
  // two pinches add roll (Z). Hats/clothing keep the translate gesture.
  const isExhibit = calibSlot === "rightHand" || calibSlot === "leftHand";

  // ---- Single pinch: translate (hats/clothing) OR rotate X/Y (exhibits) ----
  if (active.length === 1) {
    const wantMode = isExhibit ? "rotate-xy" : "translate";
    if (gestureState.mode !== wantMode) {
      gestureState.mode = wantMode;
      gestureState.startPinches = [{ x: active[0].x, y: active[0].y }];
      gestureState.startConfig = {
        offset: [...(att.config.offset || [0, 0, 0])],
        rotation: [...(att.config.rotation || [0, 0, 0])],
      };
      return;
    }
    const dx = active[0].x - gestureState.startPinches[0].x;
    const dy = active[0].y - gestureState.startPinches[0].y;
    const xMul = scene.mirror ? -1 : 1;

    if (isExhibit) {
      // Drag horizontally → yaw (Y); drag vertically → pitch (X).
      att.config.rotation = [
        gestureState.startConfig.rotation[0] + dy * GESTURE_ROTATE_SENS,
        gestureState.startConfig.rotation[1] + dx * GESTURE_ROTATE_SENS * xMul,
        gestureState.startConfig.rotation[2],
      ];
    } else {
      att.config.offset = [
        gestureState.startConfig.offset[0] + dx * GESTURE_TRANSLATE_SENS * xMul,
        gestureState.startConfig.offset[1] - dy * GESTURE_TRANSLATE_SENS, // image-Y is down
        gestureState.startConfig.offset[2],
      ];
    }
    att.applyConfig();
    syncCalibSliders();
    return;
  }

  // ---- Scale + Rotate Z (two pinches) ----
  const p1 = active[0], p2 = active[1];
  const dist  = Math.hypot(p2.x - p1.x, p2.y - p1.y);
  const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);

  if (gestureState.mode !== "scale-rotate") {
    // If pinches are too close at start, refuse to lock in — avoids huge
    // scale ratios from noisy near-zero starting distances.
    if (dist < GESTURE_TWO_HAND_MIN_DIST) return;
    gestureState.mode = "scale-rotate";
    gestureState.startDistance = dist;
    gestureState.startAngle    = angle;
    gestureState.startConfig = {
      scale: att.config.scale ?? 1.0,
      rotation: [...(att.config.rotation || [0, 0, 0])],
    };
    return;
  }
  const distRatio = dist / gestureState.startDistance;
  let dAngle = angle - gestureState.startAngle;
  // normalise to (-π, π]
  dAngle = Math.atan2(Math.sin(dAngle), Math.cos(dAngle));
  const dAngleDeg = (dAngle * 180) / Math.PI;
  const angleSign = scene.mirror ? -1 : 1;

  att.config.scale = Math.max(GESTURE_SCALE_MIN, gestureState.startConfig.scale * distRatio);
  att.config.rotation = [
    gestureState.startConfig.rotation[0],
    gestureState.startConfig.rotation[1],
    gestureState.startConfig.rotation[2] + dAngleDeg * angleSign,
  ];
  att.applyConfig();
  syncCalibSliders();
}

// ---------- Render loop ----------

function loop(t) {
  frames++;
  if (t - lastFpsT > 500) {
    const fps = Math.round((frames * 1000) / (t - lastFpsT));
    fpsEl.textContent = `${fps} fps`;
    frames  = 0;
    lastFpsT = t;
  }

  // Skip detectors when no item in that anchor is selected — biggest mobile win.
  // In calibration mode keep both on so the skeleton ghost remains useful even
  // before the user picks an item.
  const needPose  = !!(selected.hat || selected.top) || calibrationMode;
  // In try-on with a dress, track hands too so the skin overlay can occlude the
  // dress only where a hand actually crosses in front of it.
  const needHands = !!(selected.rightHand || selected.leftHand) || calibrationMode ||
                    (tryonEnabled && !!selected.top);
  const { pose, worldPose, hands, handedness } =
    tracker.detect(video, t, { needPose, needHands });
  lastPose  = pose;
  lastHands = hands;

  if (pose) {
    if (selected.hat) {
      // anchor: "helmet" wraps the head (sholom/kaska); default "hat" sits above
      const att = attachments.get(selected.hat);
      const fn  = att.config.anchor === "helmet" ? computeHelmet : computeHead;
      const r = fn(pose, worldPose, scene);
      if (r) att.apply(r.position, r.scale, r.quaternion, t, r.edge);
    }
    if (selected.top) {
      const r = computeTorso(pose, worldPose, scene);
      if (r) attachments.get(selected.top).apply(r.position, r.scale, r.quaternion, t, r.edge);
    }
  }

  for (let i = 0; i < hands.length; i++) {
    const handLm = hands[i];
    const label  = handedness[i]?.[0]?.categoryName ?? "Right";
    // MediaPipe HandLandmarker is trained on selfie (mirrored) frames. In
    // back-camera mode the input isn't pre-mirrored, so the labels come out
    // inverted — what the model calls "Right" is actually the subject's left.
    const isUserRight = scene.mirror ? (label === "Right") : (label === "Left");
    const itemId = isUserRight ? selected.rightHand : selected.leftHand;
    if (!itemId) continue;
    const r = computeHand(handLm, scene);
    if (r) attachments.get(itemId).apply(r.position, r.scale, r.quaternion, t);
  }

  // Gesture-driven editing — runs only when calibrationMode is on.
  if (calibrationMode) updateGesture(hands);
  else if (gestureState.mode !== "idle") resetGesture();

  drawSkeleton();
  if (tryonEnabled) {
    renderRealistic(t);
  } else {
    scene.camera.layers.enableAll();
    scene.render();
  }
  requestAnimationFrame(loop);
}

init();
