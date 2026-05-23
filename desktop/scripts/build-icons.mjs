// Generate app icon raster assets from the master SVG.
//
// Inputs:
//   desktop/src/renderer/assets/app-icon.svg  (source of truth)
//
// Outputs:
//   assets/app_icon.png       (1024x1024 master PNG)
//   assets/app_icon_<n>.png   (256, 128, 64, 48, 32, 16)
//   assets/app_icon.ico       (Windows multi-resolution: 256/128/64/48/32/16)
//
// Usage:
//   npm run build:icons   (from desktop/)

import path from "node:path";
import { mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import pngToIco from "png-to-ico";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "..", "..");
const srcSvg = path.join(projectRoot, "desktop", "src", "renderer", "assets", "app-icon.svg");
const outDir = path.join(projectRoot, "assets");

const SIZES = [1024, 512, 256, 128, 64, 48, 32, 16];
const ICO_SIZES = [256, 128, 64, 48, 32, 16];

async function main() {
  const svg = await readFile(srcSvg);
  await mkdir(outDir, { recursive: true });

  // Generate PNGs at every size in SIZES.
  const pngBuffers = new Map();
  for (const size of SIZES) {
    const buffer = await sharp(svg, { density: Math.max(72, size * 1.5) })
      .resize(size, size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png({ compressionLevel: 9 })
      .toBuffer();
    pngBuffers.set(size, buffer);
    if (size === 1024) {
      await writeFile(path.join(outDir, "app_icon.png"), buffer);
      console.log(`  wrote app_icon.png        (${size}x${size}, ${(buffer.length / 1024).toFixed(1)} KB)`);
    } else {
      const filename = `app_icon_${size}.png`;
      await writeFile(path.join(outDir, filename), buffer);
      console.log(`  wrote ${filename.padEnd(22)}(${size}x${size}, ${(buffer.length / 1024).toFixed(1)} KB)`);
    }
  }

  // Build a Windows .ico containing the resolutions Windows actually uses.
  const icoBuffers = ICO_SIZES.map((size) => pngBuffers.get(size)).filter(Boolean);
  const icoBuffer = await pngToIco(icoBuffers);
  await writeFile(path.join(outDir, "app_icon.ico"), icoBuffer);
  console.log(
    `  wrote app_icon.ico        (${ICO_SIZES.join("/")} multi-res, ${(icoBuffer.length / 1024).toFixed(1)} KB)`
  );

  // Best-effort cleanup of legacy artifacts that conflicted with the new pipeline.
  // We keep this idempotent: just no-op if missing.
  for (const stale of ["extracted_exe_icon.png", "extracted_exe_icon_icon2.png"]) {
    const stalePath = path.join(outDir, stale);
    try {
      await rm(stalePath);
      console.log(`  removed legacy            ${stale}`);
    } catch (err) {
      if (err && err.code !== "ENOENT") throw err;
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
