// Attachments — compute target transforms from MediaPipe landmarks and
// drive a Three.js group through per-channel One Euro filtering.
//
// Three anchor types:
//   - head:        hat / headwear, aligned to ear line + nose direction
//   - torso:       upper-body clothing, aligned to shoulder line + spine
//   - left_hand /
//     right_hand:  museum exhibit attached to a palm
//
// Per-item config (from items.json) can adjust:
//   offset:   [x, y, z] in local model space (units of model)
//   rotation: [x, y, z] degrees, applied on top of the computed orientation
//   scale:    multiplier on top of the computed scale

import * as THREE from "three";
import { OneEuroFilter } from "./filter.js";

// MediaPipe Pose Landmarker — index map (33 points)
export const POSE = {
  NOSE: 0,
  LEFT_EYE_INNER: 1, LEFT_EYE: 2, LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4, RIGHT_EYE: 5, RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  MOUTH_LEFT: 9, MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_PINKY: 17, RIGHT_PINKY: 18,
  LEFT_INDEX: 19, RIGHT_INDEX: 20,
  LEFT_THUMB: 21, RIGHT_THUMB: 22,
  LEFT_HIP: 23, RIGHT_HIP: 24,
};

// MediaPipe Hand Landmarker — index map (21 points per hand)
export const HAND = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

const VISIBILITY_THRESHOLD = 0.5;

function visible(lm) {
  // visibility may be undefined for hand landmarks — treat as visible
  return lm && (lm.visibility === undefined || lm.visibility > VISIBILITY_THRESHOLD);
}

// ---------- Attachment ----------

// Three.js render layers used by the realistic try-on compositing pipeline.
// Each Attachment puts its 3D node on one of these layers; the main loop
// renders each layer to a separate canvas and composes them with the
// segmenter's per-pixel class mask.
export const LAYER = {
  HATS:       0,  // hats, helmets — composited above hair
  TOPS:       1,  // virtual clothing — composited above the erased real clothes
  HAND_ITEMS: 2,  // exhibits in hand — composited BELOW skin so the hand wraps the grip
};

export class Attachment {
  constructor(scene, config, slot = "hat") {
    this.scene = scene;
    this.config = config;
    this.slot = slot; // "hat" | "top" | "hand"
    this.group = new THREE.Group();
    this.model = null;
    this.visible = false;

    // Per-channel One Euro filters: 3 position + 1 scale + 4 quaternion
    this.posF = [new OneEuroFilter(), new OneEuroFilter(), new OneEuroFilter()];
    this.scaleF = new OneEuroFilter();
    this.quatF = [
      new OneEuroFilter(), new OneEuroFilter(),
      new OneEuroFilter(), new OneEuroFilter(),
    ];

    // Inner group lets us bake per-item offset/rotation/scale without
    // touching the outer transform driven by tracking.
    this.inner = new THREE.Group();
    this.group.add(this.inner);
  }

  async load() {
    // Optional mtl override (for .obj). For .glb/.gltf/.fbx this is ignored.
    const raw = await this.scene.loadModel(this.config.file, {
      mtl: this.config.mtl,
    });

    // ----- OPTIONAL SUB-MESH FILTER -----
    // Some .glb files are assembled bundles (e.g. "Obladunok" contains both
    // helmet and breastplate). The user can pick parts to keep / drop via
    // items.json:
    //   "parts":        ["regex", ...]  // whitelist — keep only matching nodes
    //   "excludeParts": ["regex", ...]  // blacklist — remove matching nodes
    // Names of mesh nodes are logged to the console so you can discover what
    // to filter without opening Blender.
    this._filterParts(raw, this.config);

    // ----- AUTO-FIT NORMALIZATION -----
    // Models from Sketchfab / Blender export at wildly different scales
    // (mm vs m, 100x, etc) and often have an off-origin pivot. Normalize
    // every model to fit inside a 1-unit cube centered at origin. The
    // tracking layer then multiplies by headSize / shoulderWidth so the
    // model ends up at human scale regardless of how the .glb was exported.
    //
    // Optional config.fitIgnoreParts (regex list): meshes matching any
    // pattern are STILL RENDERED but ignored when computing the fit box.
    // Use this for tall ornaments (helmet spikes, plumes, antennas) that
    // would otherwise pull the bbox centre away from the model's body
    // and make the anchor formula misalign.
    const fitIgnore = (this.config.fitIgnoreParts || []).map((p) => new RegExp(p, "i"));
    const box = new THREE.Box3();
    let included = 0;
    raw.traverse((n) => {
      if (!n.isMesh) return;
      if (fitIgnore.length && fitIgnore.some((rx) => rx.test(n.name || ""))) return;
      box.expandByObject(n);
      included++;
    });
    if (included === 0) box.setFromObject(raw); // fallback: use everything

    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    // Normalisation reference dimension:
    //   tops → WIDTH (size.x). The tracking scale then sets the garment's width
    //          to the shoulder span and the height follows the model's own
    //          aspect ratio — realistic (a dress is as wide as the shoulders),
    //          instead of scaling by height and letting the width blow out.
    //   others → max dimension (helmets, hand props).
    const refDim = (this.slot === "top" ? size.x : maxDim) || maxDim;
    const fit = 1.0 / refDim;

    // Longest local axis + its half-extent (in the chosen normalised units) —
    // used by seatGrip to seat a weapon's grip into the palm.
    this.longAxis = new THREE.Vector3(0, 1, 0);
    if (size.x >= size.y && size.x >= size.z) this.longAxis.set(1, 0, 0);
    else if (size.z >= size.y && size.z >= size.x) this.longAxis.set(0, 0, 1);
    this.longHalf = (maxDim / 2) / refDim;

    // Vertical half-extent (local +Y) in the chosen normalised units. Lets
    // apply() align a model EDGE to an anchor — a dress's TOP to the shoulder
    // line, a helmet's BOTTOM rim to mid-forehead — using the model's real
    // proportions. Assumes upright authoring (Y = height). pivotY = 0 for
    // edge-aligned items (edge-align supersedes pivotY).
    this.normHalfY = (size.y / 2) / refDim;

    // Center the raw model at origin
    raw.position.sub(center);

    // Optional config.pivotY in [-1, +1] shifts the pivot away from bbox center
    // along the model's Y axis. Useful when bbox center isn't a good anchor:
    //   pivotY = -0.5  → pivot near bbox bottom → model appears HIGHER in world
    //                    (helmet rim becomes the anchor, body extends up onto head)
    //   pivotY =  0    → pivot at bbox center (default)
    //   pivotY = +0.5  → pivot near bbox top → model appears LOWER in world
    // This avoids needing to identify and ignore specific mesh parts when a
    // helmet has a tall spike / plume that pulls the bbox upward.
    const pivotY = this.config.pivotY ?? 0;
    if (pivotY !== 0) {
      raw.position.y -= (pivotY * size.y) / 2;
    }

    // Wrap in a normalizer node (scale-to-unit, fixed at load)
    const normalizer = new THREE.Group();
    normalizer.scale.set(fit, fit, fit);
    normalizer.add(raw);

    // `this.model` is the user-controlled transform (offset/rotation/scale
    // from items.json / calibration UI). All applyConfig() writes go here.
    this.model = new THREE.Group();
    this.model.add(normalizer);

    this.inner.add(this.model);
    this.applyConfig();
  }

  // Drop child nodes from `root` based on optional whitelist / blacklist of
  // regex strings in items.json. Always logs the full mesh-name list so the
  // user can see what is available to filter on.
  _filterParts(root, cfg) {
    // Log mesh names so users can find them for fitIgnoreParts / excludeParts
    // configuration (DevTools console only — UI was removed).
    const names = [];
    root.traverse((n) => { if (n.isMesh && n.name) names.push(n.name); });
    if (names.length > 0) {
      console.info(`[${cfg.id}] меші у моделі:`, names);
    }
    const include = (cfg.parts        || []).map((p) => new RegExp(p, "i"));
    const exclude = (cfg.excludeParts || []).map((p) => new RegExp(p, "i"));
    if (include.length === 0 && exclude.length === 0) return;

    const toRemove = [];
    root.traverse((n) => {
      if (!n.isMesh || !n.name) return;
      const keep =
        (include.length === 0 || include.some((rx) => rx.test(n.name))) &&
        !exclude.some((rx) => rx.test(n.name));
      if (!keep) toRemove.push(n);
    });
    for (const n of toRemove) {
      n.parent?.remove(n);
      n.geometry?.dispose?.();
      const mats = Array.isArray(n.material) ? n.material : (n.material ? [n.material] : []);
      for (const m of mats) m.dispose?.();
    }
  }

  // Re-apply config (offset / rotation / scale) to the model node immediately.
  // Called by the calibration UI after slider changes.
  applyConfig() {
    if (!this.model) return;
    const offset = this.config.offset ?? [0, 0, 0];
    const rotation = this.config.rotation ?? [0, 0, 0];
    const baseScale = this.config.scale ?? 1.0;
    this.model.position.set(offset[0], offset[1], offset[2]);
    this.model.rotation.set(
      THREE.MathUtils.degToRad(rotation[0]),
      THREE.MathUtils.degToRad(rotation[1]),
      THREE.MathUtils.degToRad(rotation[2])
    );
    this.model.scale.set(baseScale, baseScale, baseScale);
    // Clear filter history so position snaps to the new config without lag.
    for (const f of this.posF) { f.xFilter.y = null; f.xFilter.s = null; f.lastTime = null; }
    this.scaleF.xFilter.y = null; this.scaleF.xFilter.s = null; this.scaleF.lastTime = null;
    for (const f of this.quatF) { f.xFilter.y = null; f.xFilter.s = null; f.lastTime = null; }
  }

  show() {
    if (!this.visible && this.model) {
      this.scene.scene.add(this.group);
      // Assign render layer based on slot so the 3-pass compositor can
      // render each category separately. Layers don't propagate to children.
      const layer =
        this.slot === "top"  ? LAYER.TOPS :
        this.slot === "hand" ? LAYER.HAND_ITEMS :
                               LAYER.HATS;
      this.group.traverse((c) => c.layers.set(layer));
      this.visible = true;
    }
  }

  hide() {
    if (this.visible) {
      this.scene.scene.remove(this.group);
      this.visible = false;
    }
  }

  // edge: null | "top" | "bottom" — align that edge of the model to `pos`
  // (the anchor point), instead of the model's centre. Used so a dress hangs
  // from the shoulder line and a helmet's rim sits at mid-forehead.
  apply(pos, scale, quat, t, edge = null) {
    // Smooth each channel independently — One Euro keeps quick motion responsive
    // while removing high-frequency jitter on stationary positions.
    const x = this.posF[0].filter(pos.x, t);
    const y = this.posF[1].filter(pos.y, t);
    const z = this.posF[2].filter(pos.z, t);
    this.group.position.set(x, y, z);

    const s = this.scaleF.filter(scale, t);
    this.group.scale.set(s, s, s);

    const qx = this.quatF[0].filter(quat.x, t);
    const qy = this.quatF[1].filter(quat.y, t);
    const qz = this.quatF[2].filter(quat.z, t);
    const qw = this.quatF[3].filter(quat.w, t);
    this.group.quaternion.set(qx, qy, qz, qw).normalize();

    // Edge alignment: move the model so its TOP or BOTTOM edge lands exactly on
    // the anchor point. halfH is the model's world half-height along its local
    // +Y (config rotation + tracking orientation applied).
    if (edge && this.normHalfY != null) {
      const upWorld = new THREE.Vector3(0, 1, 0)
        .applyQuaternion(this.model.quaternion)
        .applyQuaternion(this.group.quaternion);
      const halfH = s * (this.config.scale ?? 1) * this.normHalfY;
      // top edge → move model DOWN by halfH; bottom edge → move UP by halfH.
      const sign = edge === "top" ? -1 : 1;
      this.group.position.add(upWorld.multiplyScalar(sign * halfH));
    }

    // seatGrip: shift the anchor outward along the model's longest axis by half
    // its length so the grip END sits in the palm and the blade extends out.
    // `gripFlip` chooses which end is the grip. Works for any model orientation
    // because the long axis is transformed by both the config rotation and the
    // tracking orientation.
    if (this.config.seatGrip && this.longAxis) {
      const dir = this.longAxis.clone()
        .applyQuaternion(this.model.quaternion)   // config rotation
        .applyQuaternion(this.group.quaternion);  // tracking orientation
      const halfLen = s * (this.config.scale ?? 1) * (this.longHalf ?? 0.5);
      const sign = this.config.gripFlip ? -1 : 1;
      this.group.position.add(dir.multiplyScalar(sign * halfLen));
    }
  }
}

// ---------- Anchor-specific transform computation ----------

// Convert a MediaPipe worldLandmark (metres; x = image-right, y = down,
// z = away from camera) into our NON-MIRRORED scene frame (x right, y up,
// z toward camera). This is a proper rotation of the frame, so a basis built
// from these vectors is right-handed.
function _wl(lm) {
  return new THREE.Vector3(lm.x, -lm.y, -lm.z);
}

// Mirror-correct a quaternion that was built in the non-mirrored frame so it
// matches the on-screen (x-flipped) selfie view. Reflecting a rotation across
// the x-plane maps q = (x,y,z,w) → (x,-y,-z,w). Without this, yaw/roll appear
// reversed in mirror mode — the "model turns the wrong way" bug.
function _mirrorQuat(q) {
  q.set(q.x, -q.y, -q.z, q.w);
  return q;
}

// Flip x of a direction vector so it can be used for positioning in the
// x-flipped scene (mirror mode).
function _mirrorVec(v) {
  v.x = -v.x;
  return v;
}

// Foreshortening compensation. When the subject turns sideways, the line
// between two landmarks projects shorter on screen even though its real length
// is unchanged — so any scale derived from the on-screen distance shrinks.
// worldLandmarks give the true 3D vector; the ratio (full length / its screen
// projection) ≈ 1/cos(turn angle). Multiply the on-screen size by this ratio to
// keep the model's scale stable through a turn. Capped so it can't explode near
// a full profile where the projection approaches zero.
function _foreshortenRatio(wA, wB) {
  const dx = wA.x - wB.x, dy = wA.y - wB.y, dz = wA.z - wB.z;
  const proj = Math.hypot(dx, dy);
  if (proj < 1e-4) return 3;
  return Math.min(Math.hypot(dx, dy, dz) / proj, 3);
}

// Shared head basis — used by both hat and helmet anchors.
//
// POSITION + SCALE: image landmarks (placed where the head appears on screen).
// ORIENTATION:      worldLandmarks when available — true 3D, so the basis stays
//                   valid even at extreme side angles where the image ears
//                   project onto the same point. The quaternion is built in the
//                   non-mirrored frame, then mirror-corrected for the display.
//                   Falls back to the image basis (already scene-space) if no
//                   worldPose.
function _headBasis(pose, worldPose, scene) {
  const lEar = pose[POSE.LEFT_EAR];
  const rEar = pose[POSE.RIGHT_EAR];
  const nose = pose[POSE.NOSE];
  if (!visible(lEar) || !visible(rEar) || !visible(nose)) return null;

  const lEarW = scene.imageToWorld(lEar.x, lEar.y, lEar.z * 0.5);
  const rEarW = scene.imageToWorld(rEar.x, rEar.y, rEar.z * 0.5);
  const noseW = scene.imageToWorld(nose.x, nose.y, nose.z * 0.5);

  const earMid = new THREE.Vector3().addVectors(lEarW, rEarW).multiplyScalar(0.5);
  // headSize from the on-screen ear distance, compensated for head rotation via
  // worldLandmarks so the helmet keeps its size when the head turns sideways.
  let headSize = Math.hypot(rEarW.x - lEarW.x, rEarW.y - lEarW.y);
  if (worldPose && worldPose[POSE.LEFT_EAR] && worldPose[POSE.RIGHT_EAR]) {
    headSize *= _foreshortenRatio(worldPose[POSE.LEFT_EAR], worldPose[POSE.RIGHT_EAR]);
  }
  headSize = Math.max(headSize, 0.05);

  let quat, up;
  const wlR = worldPose && worldPose[POSE.RIGHT_EAR];
  const wlL = worldPose && worldPose[POSE.LEFT_EAR];
  const wlN = worldPose && worldPose[POSE.NOSE];
  if (wlR && wlL && wlN) {
    // Right-handed basis in the non-mirrored frame, then mirror-correct.
    const a = _wl(wlL), b = _wl(wlR), n = _wl(wlN);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const xAxis = new THREE.Vector3().subVectors(a, b).normalize();
    const forward = new THREE.Vector3().subVectors(n, mid).normalize();
    const yAxis = new THREE.Vector3().crossVectors(forward, xAxis).normalize();
    const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
    const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
    quat = new THREE.Quaternion().setFromRotationMatrix(m);
    up = yAxis.clone();
    if (scene.mirror) { _mirrorQuat(quat); _mirrorVec(up); }
  } else {
    // Image-only fallback (already in mirrored scene space → no correction).
    const xAxis = new THREE.Vector3().subVectors(rEarW, lEarW).normalize();
    if (!scene.mirror) xAxis.negate();
    const forward = new THREE.Vector3().subVectors(noseW, earMid).normalize();
    const yAxis = new THREE.Vector3().crossVectors(forward, xAxis).normalize();
    const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
    const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
    quat = new THREE.Quaternion().setFromRotationMatrix(m);
    up = yAxis.clone();
  }

  return { earMid, headSize, up, quat };
}

// HAT: the model's BOTTOM edge rests on the crown of the head (caps, wreaths).
// Anchor = head top (≈ 1.0·headSize above the ear line); the hat sits on it.
export function computeHead(pose, worldPose, scene) {
  const b = _headBasis(pose, worldPose, scene);
  if (!b) return null;
  const crown = b.earMid.clone().add(b.up.clone().multiplyScalar(b.headSize * 1.0));
  return { position: crown, scale: b.headSize * 2.2, quaternion: b.quat, edge: "bottom" };
}

// HELMET: the model's BOTTOM edge (rim) sits at mid-forehead, and the helmet
// extends upward over the skull from there. Anchor = mid-forehead ≈ 0.4·headSize
// above the ear line. No manual offset needed — apply() seats the rim using the
// model's real height.
export function computeHelmet(pose, worldPose, scene) {
  const b = _headBasis(pose, worldPose, scene);
  if (!b) return null;
  const foreheadMid = b.earMid.clone().add(b.up.clone().multiplyScalar(b.headSize * 0.4));
  return { position: foreheadMid, scale: b.headSize * 1.6, quaternion: b.quat, edge: "bottom" };
}


// TORSO: an UPRIGHT garment that YAWS with the body.
//   - "Up" is locked to world-vertical, so the dress never tilts or rides up —
//     including when you turn your back (the old full-3D basis flipped/​rose).
//   - The facing direction (yaw) comes from the shoulder line's HORIZONTAL
//     orientation in 3D: facing camera → front of dress; back to camera → back;
//     sideways → side. So it turns with you, just stays vertical.
//
// POSITION: shoulder line, raised to the neck base. SCALE: shoulder width,
// foreshorten-compensated so it stays constant through a turn.
export function computeTorso(pose, worldPose, scene) {
  const lShoulder = pose[POSE.LEFT_SHOULDER];
  const rShoulder = pose[POSE.RIGHT_SHOULDER];
  if (!visible(lShoulder) || !visible(rShoulder)) return null;

  const lShW = scene.imageToWorld(lShoulder.x, lShoulder.y, 0);
  const rShW = scene.imageToWorld(rShoulder.x, rShoulder.y, 0);
  const shMid = new THREE.Vector3().addVectors(lShW, rShW).multiplyScalar(0.5);

  // SCALE from the yaw-invariant TORSO HEIGHT (shoulders → hips). Shoulder WIDTH
  // foreshortens when you turn sideways, and MediaPipe's depth (z) is too noisy
  // to compensate reliably — that's why the dress dropped to ~0.7 on turning.
  // The vertical torso length doesn't change when you turn left/right, so the
  // scale stays constant. Falls back to (foreshorten-compensated) shoulder width
  // when hips are out of frame (e.g. a waist-up shot).
  const lHip = pose[POSE.LEFT_HIP];
  const rHip = pose[POSE.RIGHT_HIP];
  let scale;
  if (visible(lHip) && visible(rHip)) {
    const lHipW = scene.imageToWorld(lHip.x, lHip.y, 0);
    const rHipW = scene.imageToWorld(rHip.x, rHip.y, 0);
    const hipMid = new THREE.Vector3().addVectors(lHipW, rHipW).multiplyScalar(0.5);
    // 0.8 maps average shoulder-to-hip length to the previous shoulder-width
    // scale; tune per-model with config.scale in items.json.
    scale = shMid.distanceTo(hipMid) * 0.8;
  } else {
    let sw = Math.hypot(rShW.x - lShW.x, rShW.y - lShW.y);
    if (worldPose && worldPose[POSE.LEFT_SHOULDER] && worldPose[POSE.RIGHT_SHOULDER]) {
      sw *= _foreshortenRatio(worldPose[POSE.LEFT_SHOULDER], worldPose[POSE.RIGHT_SHOULDER]);
    }
    scale = Math.max(sw, 0.10) * 1.3;
  }

  let quat;
  if (worldPose && worldPose[POSE.LEFT_SHOULDER] && worldPose[POSE.RIGHT_SHOULDER]) {
    // Upright-yaw basis built in the non-mirrored frame, then mirror-corrected.
    const a = _wl(worldPose[POSE.LEFT_SHOULDER]);
    const b = _wl(worldPose[POSE.RIGHT_SHOULDER]);
    // Horizontal component of the shoulder line only → pure yaw, no tilt/rise.
    const xAxis = new THREE.Vector3(a.x - b.x, 0, a.z - b.z);
    if (xAxis.lengthSq() < 1e-6) xAxis.set(1, 0, 0);
    xAxis.normalize();
    const yAxis = new THREE.Vector3(0, 1, 0);
    const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
    const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
    quat = new THREE.Quaternion().setFromRotationMatrix(m);
    if (scene.mirror) _mirrorQuat(quat);
  } else {
    // Image fallback: frontal billboard with shoulder-line roll.
    const xF = new THREE.Vector3(rShW.x - lShW.x, rShW.y - lShW.y, 0).normalize();
    const zF = new THREE.Vector3(0, 0, 1);
    const yF = new THREE.Vector3().crossVectors(zF, xF).normalize();
    quat = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(xF, yF, zF)
    );
  }

  // Up is world-vertical, so raise straight up to the neck base; apply() seats
  // the model's TOP edge there so the dress hangs from the shoulders. The raise
  // is tied to `scale` (also yaw-invariant) so the collar stays put on turning.
  const anchor = shMid.clone().add(new THREE.Vector3(0, scale * 0.14, 0));
  return { position: anchor, scale, quaternion: quat, edge: "top" };
}

// HAND: anchor a prop to the palm.
//
// The previous version oriented the prop by the palm normal, which is computed
// from noisy hand landmarks (especially depth) and made the prop "swim". This
// version keeps the prop in a stable, camera-facing plane and only rotates it
// in 2D to follow the hand's on-screen direction — far steadier and easy to
// calibrate. The grip is seated into the palm in Attachment.apply() when the
// item has `seatGrip` set.
export function computeHand(handLm, scene) {
  const wrist = handLm[HAND.WRIST];
  const middleMcp = handLm[HAND.MIDDLE_MCP];
  if (!wrist || !middleMcp) return null;

  // Ignore landmark depth (z) — it's the noisiest channel and the main cause
  // of the prop flailing. Work purely in the screen plane.
  const wristW = scene.imageToWorld(wrist.x, wrist.y, 0);
  const middleW = scene.imageToWorld(middleMcp.x, middleMcp.y, 0);

  const palmCenter = new THREE.Vector3().addVectors(wristW, middleW).multiplyScalar(0.5);
  const handSize = Math.max(wristW.distanceTo(middleW), 0.04);
  // Props read better noticeably larger than the palm (a sword is ~7x a palm).
  // Per-item config.scale fine-tunes on top of this.
  const scale = handSize * 2.4;

  // Pointing axis = wrist → middle-finger MCP, projected to the screen plane.
  // The prop's local +Y is aligned to this; +Z faces the camera.
  const yAxis = new THREE.Vector3(middleW.x - wristW.x, middleW.y - wristW.y, 0).normalize();
  const zRef = new THREE.Vector3(0, 0, 1);
  const xAxis = new THREE.Vector3().crossVectors(yAxis, zRef).normalize();
  const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();

  const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
  const quat = new THREE.Quaternion().setFromRotationMatrix(m);

  return { position: palmCenter, scale, quaternion: quat };
}
