// Scene — Three.js setup with a perspective camera aligned to the webcam frame
// so MediaPipe image coordinates map cleanly to a near-camera plane.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";

export class Scene {
  constructor(canvas, width, height) {
    // On mobile devicePixelRatio is often 2-3, which means a 1280x720 video
    // becomes a 2560x1440+ render target — kills fps on integrated GPUs.
    // Cap to 1.5; the difference is barely visible on a phone display
    // because the canvas is overlaid on the camera frame anyway.
    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: !isMobile,             // MSAA is expensive on mobile GPUs
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,      // required for photo capture via drawImage
    });
    this.renderer.setSize(width, height, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 1.25 : 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();

    // Selfie/mirror mode: true for front camera (image is displayed mirrored
    // so we flip X in imageToWorld to compensate); false for back camera
    // (no display mirror, X passes through). Switched at runtime when the
    // user toggles the camera.
    this.mirror = true;

    this.fov = 50;
    this.aspect = width / height;
    this.camera = new THREE.PerspectiveCamera(this.fov, this.aspect, 0.01, 100);
    this.camera.position.set(0, 0, 2);

    // Curatorial lighting — soft fill + warm key + cool rim.
    // enableAll() because the compositor renders each layer separately;
    // without it, objects on non-default layers would render unlit.
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    ambient.layers.enableAll();
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(0.5, 1.2, 1.5);
    key.layers.enableAll();
    this.scene.add(key);

    const warm = new THREE.DirectionalLight(0xffd9a8, 0.35);
    warm.position.set(-1.2, 0.4, 1);
    warm.layers.enableAll();
    this.scene.add(warm);

    const rim = new THREE.DirectionalLight(0x9ab4ff, 0.25);
    rim.position.set(0, 0.3, -1);
    rim.layers.enableAll();
    this.scene.add(rim);

    // GLTF (+ DRACO compression). GLTFLoader resolves external textures
    // relative to the .gltf file URL automatically.
    const draco = new DRACOLoader();
    draco.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.6/");
    this.gltfLoader = new GLTFLoader();
    this.gltfLoader.setDRACOLoader(draco);

    // FBX — single-file binary with embedded textures. (OBJ/MTL loaders are
    // created per-load in _loadOBJ since they need per-file resource paths.)
    this.fbxLoader = new FBXLoader();
  }

  setSize(width, height) {
    this.renderer.setSize(width, height, false);
    this.aspect = width / height;
    this.camera.aspect = this.aspect;
    this.camera.updateProjectionMatrix();
  }

  // Convert MediaPipe image coords [0..1] to Three.js world coords.
  // imgY is flipped (top→down becomes bottom→up). X is mirrored so the
  // overlay matches the CSS-mirrored video (selfie / mirror view).
  // depthOffset (≈ MediaPipe z) shifts items along the camera axis.
  imageToWorld(imgX, imgY, depthOffset = 0) {
    const z = -depthOffset;
    const distance = this.camera.position.z - z;
    const visibleH = 2 * Math.tan((this.fov * Math.PI / 180) / 2) * distance;
    const visibleW = visibleH * this.aspect;
    const xSign = this.mirror ? -1 : 1;
    const wx = xSign * (imgX - 0.5) * visibleW;
    const wy = -(imgY - 0.5) * visibleH;
    return new THREE.Vector3(wx, wy, z);
  }

  // Format dispatch — picks a loader from the file extension.
  // Supported: .glb, .gltf (with external textures + optional DRACO),
  //            .obj (with optional .mtl + texture maps),
  //            .fbx (binary, embedded textures).
  async loadModel(url, opts = {}) {
    const ext = url.split("?")[0].split("#")[0].split(".").pop().toLowerCase();
    switch (ext) {
      case "glb":
      case "gltf":
        return this._loadGLTF(url);
      case "obj":
        return this._loadOBJ(url, opts.mtl);
      case "fbx":
        return this._loadFBX(url);
      default:
        throw new Error(`Невідомий формат моделі: .${ext}`);
    }
  }

  _loadGLTF(url) {
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(
        url,
        (gltf) => resolve(gltf.scene),
        undefined,
        (err) => reject(err)
      );
    });
  }

  // OBJ loading flow:
  //   1. If an .mtl URL is supplied (or auto-detected as <name>.mtl next to the .obj),
  //      load it first. MTLLoader resolves texture map paths relative to the .mtl URL.
  //   2. Apply the parsed materials to OBJLoader, then load the geometry.
  //   3. If no .mtl is found we fall through with a default white material.
  async _loadOBJ(url, mtlUrl) {
    const resolvedMtl = mtlUrl ?? url.replace(/\.obj($|\?)/i, ".mtl$1");
    const basePath = url.substring(0, url.lastIndexOf("/") + 1);

    let materials = null;
    try {
      materials = await new Promise((resolve, reject) => {
        const ml = new MTLLoader();
        ml.setResourcePath(basePath); // where textures live
        ml.setPath("");
        ml.load(resolvedMtl, resolve, undefined, reject);
      });
      materials.preload();
    } catch (e) {
      // No .mtl alongside .obj — OBJLoader will use the default material.
      // eslint-disable-next-line no-console
      console.warn(`MTL не знайдено для ${url} — використано матеріал за замовчуванням`);
    }

    return new Promise((resolve, reject) => {
      const ol = new OBJLoader();
      if (materials) ol.setMaterials(materials);
      ol.load(
        url,
        (obj) => {
          // Ensure textures decode in sRGB (matches the renderer outputColorSpace)
          obj.traverse((child) => {
            if (child.isMesh && child.material) {
              const mats = Array.isArray(child.material) ? child.material : [child.material];
              for (const m of mats) {
                if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
                m.needsUpdate = true;
              }
            }
          });
          resolve(obj);
        },
        undefined,
        (err) => reject(err)
      );
    });
  }

  _loadFBX(url) {
    return new Promise((resolve, reject) => {
      this.fbxLoader.load(
        url,
        (obj) => {
          obj.traverse((child) => {
            if (child.isMesh && child.material) {
              const mats = Array.isArray(child.material) ? child.material : [child.material];
              for (const m of mats) {
                if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
                m.needsUpdate = true;
              }
            }
          });
          resolve(obj);
        },
        undefined,
        (err) => reject(err)
      );
    });
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
