import { BrowserWindow, app, clipboard, dialog, ipcMain, nativeTheme, session, shell } from "electron";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..");
const desktopRoot = path.resolve(projectRoot, "desktop");

const OPEN3D_WEBRTC_URL_PATTERNS = [
  "http://127.0.0.1:8888/*",
  "http://localhost:8888/*",
  "http://[::1]:8888/*"
];
const OPEN3D_ICE_SERVER_URL_PATTERNS = [
  "http://127.0.0.1:8888/api/getIceServers*",
  "http://localhost:8888/api/getIceServers*",
  "http://[::1]:8888/api/getIceServers*"
];

let backendProcess: ChildProcessWithoutNullStreams | null = null;
let backendPort = 8765;
const backendToken = randomBytes(24).toString("hex");

async function pickPort(): Promise<number> {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (typeof address === "object" && address !== null) {
        const port = address.port;
        server.close(() => resolve(port));
        return;
      }
      server.close(() => reject(new Error("Cannot allocate local port.")));
    });
    server.on("error", reject);
  });
}

function pythonExecutable(): string {
  return path.join(projectRoot, ".venv", "Scripts", "python.exe");
}

function startBackend(): void {
  if (backendProcess !== null) {
    return;
  }

  const python = pythonExecutable();
  const pythonPath = path.join(projectRoot, "src");
  backendProcess = spawn(
    python,
    ["-m", "rail_curve_extractor.backend.app", "--host", "127.0.0.1", "--port", String(backendPort)],
    {
      cwd: projectRoot,
      env: {
        ...process.env,
        PYTHONPATH: [pythonPath, process.env.PYTHONPATH ?? ""].filter(Boolean).join(path.delimiter),
        RAIL_CURVE_BACKEND_TOKEN: backendToken
      }
    }
  );
  backendProcess.stdout.on("data", (chunk) => console.log(`[backend] ${chunk.toString().trimEnd()}`));
  backendProcess.stderr.on("data", (chunk) => console.error(`[backend] ${chunk.toString().trimEnd()}`));
  backendProcess.on("exit", (code) => {
    console.log(`[backend] exited with code ${code}`);
    backendProcess = null;
  });
}

async function configureLocalViewerSession(localBackendPort: number): Promise<void> {
  await session.defaultSession.setProxy({ mode: "direct" });
  session.defaultSession.webRequest.onBeforeRequest(
    { urls: OPEN3D_ICE_SERVER_URL_PATTERNS },
    (_details, callback) => {
      callback({
        redirectURL: `http://127.0.0.1:${localBackendPort}/open3d-local-ice`
      });
    }
  );
  session.defaultSession.webRequest.onHeadersReceived(
    { urls: OPEN3D_WEBRTC_URL_PATTERNS },
    (details, callback) => {
      const responseHeaders = { ...details.responseHeaders };
      for (const headerName of Object.keys(responseHeaders)) {
        const normalizedName = headerName.toLowerCase();
        if (normalizedName === "x-frame-options" || normalizedName === "frame-options") {
          delete responseHeaders[headerName];
        }
        if (normalizedName === "content-security-policy") {
          responseHeaders[headerName] = responseHeaders[headerName]?.filter(
            (value) => !value.toLowerCase().includes("frame-ancestors")
          );
          if (responseHeaders[headerName]?.length === 0) {
            delete responseHeaders[headerName];
          }
        }
      }
      callback({ responseHeaders });
    }
  );
}

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 1120,
    minHeight: 720,
    useContentSize: true,
    title: "Rail Curve Extractor",
    backgroundColor: nativeTheme.shouldUseDarkColors ? "#0b1120" : "#f6f7fb",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  const indexFile = path.join(desktopRoot, "dist", "index.html");
  void window.loadFile(indexFile);
}

ipcMain.handle("backend-config", () => ({
  baseUrl: `http://127.0.0.1:${backendPort}`,
  token: backendToken
}));

ipcMain.handle("dialog:open-dom", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择原始 DOM 影像",
    properties: ["openFile"],
    filters: [
      { name: "Raster Images", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:open-model", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择 DeepLab 语义分割权重",
    properties: ["openFile"],
    filters: [
      { name: "PyTorch Models", extensions: ["pt", "pth"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:open-dsm", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择 DSM 栅格",
    properties: ["openFile"],
    filters: [
      { name: "Raster Images", extensions: ["tif", "tiff"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:open-las-dir", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择 LAS/LAZ 点云目录",
    properties: ["openDirectory"]
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:select-output-dir", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择输出目录",
    properties: ["openDirectory", "createDirectory"]
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("shell:reveal", (_event, target: string) => {
  if (!target) return false;
  shell.showItemInFolder(target);
  return true;
});

ipcMain.handle("shell:open-path", async (_event, target: string) => {
  if (!target) return "";
  return await shell.openPath(target);
});

ipcMain.handle("clipboard:write", (_event, text: string) => {
  clipboard.writeText(text ?? "");
  return true;
});

app.whenReady().then(async () => {
  backendPort = await pickPort();
  await configureLocalViewerSession(backendPort);
  startBackend();
  createWindow();
});

app.on("window-all-closed", () => {
  backendProcess?.kill();
  if (process.platform !== "darwin") {
    app.quit();
  }
});
