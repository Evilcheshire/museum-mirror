// Tracker — MediaPipe Pose Landmarker + Hand Landmarker running in parallel.
// Both run on the GPU delegate where available; both run per-frame in VIDEO mode.
//
// Pose models:
//   lite  — smaller model (~3 MB), faster (~30+ fps), suitable for real-time use
//   full  — larger model (~6 MB), more accurate landmark positions and better
//            occlusion handling; noticeable on slower GPUs (may drop below 20 fps)
//
// Switch with tracker.setMode('full' | 'lite'). Detection pauses for ~1-2 s
// while the new model downloads and initialises; frames return null during that window.

// NOTE on CDN choice and %40 encoding:
//   *.ngrok-free.dev is fronted by Cloudflare with "Email Address Obfuscation"
//   enabled, which scans HTML/JS for "name@domain.tld" patterns and replaces
//   them with placeholders. The literal "[email protected]" matches and
//   gets corrupted. Encoding "@" as "%40" sidesteps Cloudflare's regex.
//
//   We use UNPKG specifically because it URL-decodes %40 back to @ when
//   resolving npm packages. jsdelivr does NOT — it treats "tasks-vision%400.10.7"
//   as a literal package name, 404s, and returns text/plain.
import {
  PoseLandmarker,
  HandLandmarker,
  ImageSegmenter,
  FilesetResolver,
} from "https://unpkg.com/@mediapipe/tasks-vision%400.10.7/vision_bundle.mjs";

const WASM_BASE =
  "https://unpkg.com/@mediapipe/tasks-vision%400.10.7/wasm";

const POSE_MODELS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
};

const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

// Multiclass selfie segmenter: 6 categories
//   0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others/accessories
const SEG_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

export const SEG_CLASS = {
  BG: 0, HAIR: 1, BODY_SKIN: 2, FACE_SKIN: 3, CLOTHES: 4, OTHERS: 5,
};

export class Tracker {
  constructor() {
    this.pose = null;
    this.hands = null;
    this.segmenter = null;
    this._vision = null;
    this._switching = false;
    this._segBusy = false;
    this.currentMode = "lite";
    // Latest multiclass mask (Uint8Array, w*h, values 0-5)
    this.lastMask = null;
  }

  async init() {
    this._vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    await this._initPose("lite");
    this.hands = await HandLandmarker.createFromOptions(this._vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  }

  async _initPose(mode) {
    if (this.pose) {
      this.pose.close();
      this.pose = null;
    }
    this.pose = await PoseLandmarker.createFromOptions(this._vision, {
      baseOptions: { modelAssetPath: POSE_MODELS[mode], delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    this.currentMode = mode;
  }

  // Lazy-init the multiclass selfie segmenter. Used by the realistic try-on
  // compositing pipeline to (a) clip clothing to the body silhouette,
  // (b) erase the user's real clothes, (c) keep hands/face above virtual layers.
  async ensureSegmenter() {
    if (this.segmenter) return;
    if (!this._vision) this._vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    this.segmenter = await ImageSegmenter.createFromOptions(this._vision, {
      baseOptions: { modelAssetPath: SEG_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
  }

  // Non-blocking segmentation: if the previous call hasn't returned we just
  // re-use the cached mask. Caps the segmenter at its natural rate without
  // stalling the render loop.
  segment(video, timestamp) {
    if (!this.segmenter || this._segBusy) return this.lastMask;
    this._segBusy = true;
    try {
      this.segmenter.segmentForVideo(video, timestamp, (result) => {
        try {
          const m = result?.categoryMask;
          if (m) {
            this.lastMask = {
              data: m.getAsUint8Array().slice(),
              width: m.width,
              height: m.height,
            };
          }
        } finally {
          result?.close?.();
          this._segBusy = false;
        }
      });
    } catch (e) {
      this._segBusy = false;
      console.warn("segmentForVideo failed:", e);
    }
    return this.lastMask;
  }

  // Switch pose model. Returns a promise; detection returns null during the switch.
  async setMode(mode) {
    if (mode === this.currentMode || this._switching) return;
    this._switching = true;
    try {
      await this._initPose(mode);
    } finally {
      this._switching = false;
    }
  }

  // Returns { pose, worldPose, hands, handedness }. Pass needPose/needHands =
  // false to skip a detector (saves CPU/GPU when that anchor is unused).
  //   pose       — 33 image landmarks (0..1), used for SCREEN POSITION
  //   worldPose  — 33 metric 3D landmarks, used for rotation-stable ORIENTATION
  //   hands      — per-hand 21 image landmarks; handedness — left/right labels
  detect(video, timestamp, { needPose = true, needHands = true } = {}) {
    if (this._switching || !this.pose || !this.hands) {
      return { pose: null, worldPose: null, hands: [], handedness: [] };
    }
    let pose = null, worldPose = null;
    let hands = [], handedness = [];
    if (needPose) {
      const r = this.pose.detectForVideo(video, timestamp);
      pose      = r.landmarks?.[0] ?? null;
      worldPose = r.worldLandmarks?.[0] ?? null;
    }
    if (needHands) {
      const r = this.hands.detectForVideo(video, timestamp);
      hands      = r.landmarks ?? [];
      handedness = r.handednesses ?? [];
    }
    return { pose, worldPose, hands, handedness };
  }
}
