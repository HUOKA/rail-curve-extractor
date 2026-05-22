import { BrowserWindow, app, dialog, ipcMain, session } from "electron";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..");
const desktopRoot = path.resolve(projectRoot, "desktop");
const MIN_WINDOW_WIDTH = 1480;
const MIN_WINDOW_HEIGHT = 900;
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
  const venvPython = path.join(projectRoot, ".venv", "Scripts", "python.exe");
  return venvPython;
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
    width: 1560,
    height: 940,
    minWidth: MIN_WINDOW_WIDTH,
    minHeight: MIN_WINDOW_HEIGHT,
    useContentSize: true,
    title: "Rail Curve Extractor",
    backgroundColor: "#f3f6f8",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  window.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT);
  window.on("resize", () => {
    const [width, height] = window.getSize();
    if (width < MIN_WINDOW_WIDTH || height < MIN_WINDOW_HEIGHT) {
      window.setSize(Math.max(width, MIN_WINDOW_WIDTH), Math.max(height, MIN_WINDOW_HEIGHT));
    }
  });

  const indexFile = path.join(desktopRoot, "dist", "index.html");
  void window.loadFile(indexFile);
}

ipcMain.handle("backend-config", () => ({
  baseUrl: `http://127.0.0.1:${backendPort}`,
  token: backendToken
}));

ipcMain.handle("dialog:open-point-cloud", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择点云文件",
    properties: ["openFile"],
    filters: [
      { name: "Point Clouds", extensions: ["las", "laz", "csv", "txt", "xyz", "npy"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("dialog:open-point-cloud-folder", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择 DJI Terra 项目文件夹或点云目录",
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
