// Photo capture — compose the mirrored webcam frame with the Three.js overlay
// into a single PNG and trigger download.

// Composite video + 3D overlay, encode as PNG, trigger download.
// Returns a Promise that resolves once the blob is created and the download
// is triggered (so callers can sequence UI feedback around the operation).
// composedOnly=true: `source` is treated as a fully-assembled image (e.g. the
// realistic try-on compositor's output) and copied as-is. mirror is ignored.
export function takePhoto(video, source, mirror = true, composedOnly = false) {
  return new Promise((resolve, reject) => {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) { reject(new Error("Камера не готова")); return; }

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");

    if (composedOnly) {
      ctx.drawImage(source, 0, 0, w, h);
    } else {
      // 1. Draw the camera frame, mirrored if the on-screen video is mirrored.
      if (mirror) {
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, w, h);
        ctx.restore();
      } else {
        ctx.drawImage(video, 0, 0, w, h);
      }
      // 2. Draw the Three.js overlay (already in mirror-aware world coords).
      ctx.drawImage(source, 0, 0, w, h);
    }

    canvas.toBlob(
      (blob) => {
        if (!blob) { reject(new Error("Не вдалося створити зображення")); return; }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `museum-mirror-${timestamp()}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        resolve();
      },
      "image/png"
    );
  });
}

function timestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    d.getFullYear() +
    pad(d.getMonth() + 1) +
    pad(d.getDate()) +
    "-" +
    pad(d.getHours()) +
    pad(d.getMinutes()) +
    pad(d.getSeconds())
  );
}
