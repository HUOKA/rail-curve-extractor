import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  Divider,
  Field,
  FluentProvider,
  Input,
  ProgressBar,
  Spinner,
  Text,
  Textarea,
  Title2,
  Title3,
  webLightTheme
} from "@fluentui/react-components";
import "./styles.css";

type BackendConfig = {
  baseUrl: string;
  token: string;
};

type HealthResponse = {
  ok: boolean;
  service: string;
  python: string;
};

type PointCloudInfo = {
  input_path: string;
  points: number;
  has_intensity: boolean;
  has_rgb: boolean;
  source_format: string;
  source_file_count?: number;
  source_paths?: string[];
  bounds: {
    minimum: number[];
    maximum: number[];
  };
};

type PreviewData = {
  input_path: string;
  input_points: number;
  sample_points: number;
  points_xy: [number, number][];
  rgb: [number, number, number][] | null;
  bounds: {
    minimum: [number, number];
    maximum: [number, number];
  };
};

type ResultOverlayTrack = {
  id: number;
  label: string;
  confidence: number;
  source: string;
  centerline_xy: [number, number][];
  rail_points_xy: [number, number][];
};

type ResultOverlayTurnout = {
  id: number;
  label: string;
  confidence: number;
  switch_point_xy: [number, number][];
  main_centerline_xy: [number, number][];
  branch_centerline_xy: [number, number][];
};

type ResultOverlay = {
  track_count: number;
  turnout_count: number;
  tracks: ResultOverlayTrack[];
  turnouts: ResultOverlayTurnout[];
  centerline_xy: [number, number][];
  rail_points_xy: [number, number][];
};

type EmbeddedOpen3DResult = {
  started: boolean;
  pid: number;
  url: string;
  max_points: number;
  full_density: boolean;
  log_path: string;
  progress_path: string;
  viewer_python: string;
  status?: EmbeddedOpen3DStatus;
};

type EmbeddedOpen3DStatus = {
  state: string;
  phase: string;
  message: string;
  current_file?: string | null;
  file_index?: number | null;
  file_count?: number | null;
  loaded_points?: number | null;
  total_points?: number | null;
  source_total_points?: number | null;
  display_points?: number | null;
  percent?: number | null;
  url: string;
  pid?: number | null;
  return_code?: number | null;
  ready: boolean;
  error?: string | null;
  updated_at?: number | null;
  log_path: string;
  progress_path: string;
};

type DomPipelineStatus = {
  job_id: string;
  state: string;
  message: string;
  stage_name?: string | null;
  stage_description?: string | null;
  stage_status?: string | null;
  stage_index?: number | null;
  stage_count?: number | null;
  percent?: number | null;
  out_dir: string;
  outputs?: Record<string, unknown>;
  summary_path?: string | null;
  error?: string | null;
  pid?: number | null;
  return_code?: number | null;
  running: boolean;
  log_path: string;
  progress_path: string;
  latest_event?: Record<string, unknown> | null;
};

type Bounds2d = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
};

type RoiBox = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  z_min: null;
  z_max: null;
};

type GuidedTrackDraft = {
  id: number;
  points: [number, number][];
  corridor_width: number;
};

type GuidedTurnoutDraft = {
  id: number;
  main_points: [number, number][];
  branch_points: [number, number][];
  corridor_width: number;
};

type GuidedTarget =
  | { kind: "track"; id: number }
  | { kind: "turnout_main"; id: number }
  | { kind: "turnout_branch"; id: number };

type ApiResult = Record<string, unknown>;
type AppPage = "data" | "annotate" | "export";
type PreviewLoadingMode = "idle" | "overview" | "focus";

const defaultOutput = "";
const OVERVIEW_PREVIEW_POINTS = 120_000;
const FOCUSED_PREVIEW_POINTS = 500_000;
const AUTO_LOD_DEBOUNCE_MS = 650;
const OPEN3D_SAFE_MAX_POINTS = 3_000_000;
const OPEN3D_VIEWER_MAX_POINTS = 12_000_000;
const OPEN3D_VIEWER_POINT_SIZE = 2;

function App() {
  if (!window.railCurve) {
    return (
      <FluentProvider theme={webLightTheme}>
        <main className="bridge-error">
          <Card className="diagnostic-card">
            <CardHeader
              header={<Title2 as="h2">Electron 桥接未加载</Title2>}
              description={<Text>preload 脚本没有成功暴露 window.railCurve，应用无法连接本地后端。</Text>}
            />
            <Text>请重新运行 `npm run build` 后再启动；如果仍然出现，查看终端中的 Electron preload 报错。</Text>
          </Card>
        </main>
      </FluentProvider>
    );
  }

  const [backend, setBackend] = useState<BackendConfig | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [inputPath, setInputPath] = useState("");
  const [domPath, setDomPath] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [dsmPath, setDsmPath] = useState("");
  const [lasDir, setLasDir] = useState("");
  const [outputDir, setOutputDir] = useState(defaultOutput);
  const [pipelineDevice, setPipelineDevice] = useState("cuda");
  const [pipelineThreshold, setPipelineThreshold] = useState("0.50");
  const [pipelineMaxTiles, setPipelineMaxTiles] = useState("0");
  const [pipelineForce, setPipelineForce] = useState(false);
  const [pipelineJobId, setPipelineJobId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<DomPipelineStatus | null>(null);
  const [configText, setConfigText] = useState("{}");
  const [pointInfo, setPointInfo] = useState<PointCloudInfo | null>(null);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [focusPreviewData, setFocusPreviewData] = useState<PreviewData | null>(null);
  const [previewLoadingMode, setPreviewLoadingMode] = useState<PreviewLoadingMode>("idle");
  const [previewLodMessage, setPreviewLodMessage] = useState("自动LOD未加载");
  const [resultOverlay, setResultOverlay] = useState<ResultOverlay | null>(null);
  const [open3dViewerUrl, setOpen3dViewerUrl] = useState<string | null>(null);
  const [open3dViewerPid, setOpen3dViewerPid] = useState<number | null>(null);
  const [open3dStatus, setOpen3dStatus] = useState<EmbeddedOpen3DStatus | null>(null);
  const [currentViewBounds, setCurrentViewBounds] = useState<Bounds2d | null>(null);
  const [resetViewToken, setResetViewToken] = useState(0);
  const [selectedRoi, setSelectedRoi] = useState<RoiBox | null>(null);
  const [roiMode, setRoiMode] = useState<"global" | "auto_tracks">("global");
  const [autoTrackCount, setAutoTrackCount] = useState(1);
  const [guidedPickEnabled, setGuidedPickEnabled] = useState(false);
  const [guidedTracks, setGuidedTracks] = useState<GuidedTrackDraft[]>([
    { id: 1, points: [], corridor_width: 5.0 }
  ]);
  const [guidedTurnouts, setGuidedTurnouts] = useState<GuidedTurnoutDraft[]>([]);
  const [activeGuidedTarget, setActiveGuidedTarget] = useState<GuidedTarget>({ kind: "track", id: 1 });
  const [activePage, setActivePage] = useState<AppPage>("data");
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<string[]>(["正在启动本地 Python 后端…"]);
  const [lastResult, setLastResult] = useState<ApiResult | null>(null);
  const overviewRequestRef = React.useRef(0);
  const focusRequestRef = React.useRef(0);
  const lastFocusBoundsKeyRef = React.useRef("");

  const canRun = useMemo(
    () => inputPath.trim().length > 0 && outputDir.trim().length > 0 && !busy,
    [busy, inputPath, outputDir]
  );
  const pipelineRunning = Boolean(pipelineStatus?.running || ["starting", "planned", "running"].includes(pipelineStatus?.state ?? ""));
  const canRunDomPipeline = useMemo(
    () => domPath.trim().length > 0 && modelPath.trim().length > 0 && outputDir.trim().length > 0 && !busy && !pipelineRunning,
    [busy, domPath, modelPath, outputDir, pipelineRunning]
  );
  const pipelineProgressValue =
    typeof pipelineStatus?.percent === "number" && Number.isFinite(pipelineStatus.percent)
      ? Math.max(0, Math.min(1, pipelineStatus.percent / 100))
      : undefined;
  const open3dLoading = Boolean(open3dStatus && ["starting", "loading", "building", "serving"].includes(open3dStatus.state));

  const guidedTargetOptions = useMemo(
    () => [
      ...guidedTracks.map((track) => ({
        key: guidedTargetToKey({ kind: "track", id: track.id }),
        label: `轨道 ${track.id} 路径点（${track.points.length}）`
      })),
      ...guidedTurnouts.flatMap((turnout) => [
        {
          key: guidedTargetToKey({ kind: "turnout_main", id: turnout.id }),
          label: `道岔 ${turnout.id} 主线（${turnout.main_points.length}）`
        },
        {
          key: guidedTargetToKey({ kind: "turnout_branch", id: turnout.id }),
          label: `道岔 ${turnout.id} 分支（${turnout.branch_points.length}）`
        }
      ])
    ],
    [guidedTracks, guidedTurnouts]
  );

  useEffect(() => {
    void window.railCurve.backendConfig().then((config) => {
      setBackend(config);
      appendLog(`后端地址：${config.baseUrl}`);
    });
  }, []);

  useEffect(() => {
    setResultOverlay(null);
    setPreviewData(null);
    setFocusPreviewData(null);
    setCurrentViewBounds(null);
    setPreviewLodMessage("自动LOD未加载");
    lastFocusBoundsKeyRef.current = "";
    setOpen3dViewerUrl(null);
    setOpen3dViewerPid(null);
    setOpen3dStatus(null);
  }, [inputPath]);

  useEffect(() => {
    setResultOverlay(null);
    setOpen3dViewerUrl(null);
    setOpen3dViewerPid(null);
    setOpen3dStatus(null);
  }, [configText]);

  useEffect(() => {
    if (!backend) {
      return;
    }
    const timer = window.setInterval(() => {
      void checkHealth(false);
    }, 1500);
    void checkHealth(true);
    return () => window.clearInterval(timer);
  }, [backend]);

  useEffect(() => {
    if (!backend || !open3dLoading) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshEmbeddedOpen3DStatus(false);
    }, 700);
    void refreshEmbeddedOpen3DStatus(false);
    return () => window.clearInterval(timer);
  }, [backend, open3dLoading, open3dStatus?.pid, open3dStatus?.state]);

  useEffect(() => {
    if (!backend || !pipelineJobId || pipelineStatus?.running === false) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshDomPipelineStatus(false);
    }, 1000);
    void refreshDomPipelineStatus(false);
    return () => window.clearInterval(timer);
  }, [backend, pipelineJobId, pipelineStatus?.running, pipelineStatus?.state]);

  useEffect(() => {
    if (
      !backend ||
      activePage !== "annotate" ||
      !inputPath.trim() ||
      !pointInfo ||
      !previewData ||
      !currentViewBounds ||
      open3dViewerUrl ||
      open3dLoading ||
      previewLoadingMode !== "idle"
    ) {
      return;
    }

    const fullBounds = boundsFromPointInfo(pointInfo);
    if (!fullBounds) {
      return;
    }
    const viewAreaRatio = boundsArea(currentViewBounds) / Math.max(boundsArea(fullBounds), 1e-6);
    const timer = window.setTimeout(() => {
      if (viewAreaRatio >= 0.55) {
        if (focusPreviewData) {
          setFocusPreviewData(null);
          setPreviewLodMessage("当前为全图低密度预览；放大后会自动加载视口高清点云。");
        }
        return;
      }

      const currentFocusBounds = focusPreviewData ? boundsFromPreview(focusPreviewData) : null;
      if (
        currentFocusBounds &&
        boundsContains(currentFocusBounds, currentViewBounds) &&
        boundsArea(currentViewBounds) / Math.max(boundsArea(currentFocusBounds), 1e-6) > 0.22
      ) {
        return;
      }

      const paddedBounds = expandBounds(currentViewBounds, 0.75, fullBounds);
      const nextBoundsKey = boundsKey(paddedBounds);
      if (lastFocusBoundsKeyRef.current === nextBoundsKey) {
        return;
      }
      lastFocusBoundsKeyRef.current = nextBoundsKey;
      void loadPreviewLayer("focus", paddedBounds, FOCUSED_PREVIEW_POINTS, false);
    }, AUTO_LOD_DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [
    activePage,
    backend,
    currentViewBounds,
    focusPreviewData,
    inputPath,
    open3dLoading,
    open3dViewerUrl,
    pointInfo,
    previewData,
    previewLoadingMode
  ]);

  function appendLog(message: string) {
    const time = new Date().toLocaleTimeString();
    setLogs((items) => [`[${time}] ${message}`, ...items].slice(0, 120));
  }

  async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
    if (!backend) {
      throw new Error("后端尚未就绪。");
    }
    const response = await fetch(`${backend.baseUrl}${path}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        "x-local-token": backend.token,
        ...(init.headers ?? {})
      }
    });
    const payload = (await response.json()) as T & { detail?: string };
    if (!response.ok) {
      throw new Error(payload.detail ?? `请求失败：${response.status}`);
    }
    return payload;
  }

  async function checkHealth(verbose: boolean) {
    try {
      const data = await apiRequest<HealthResponse>("/api/health");
      setHealth(data);
      if (verbose) {
        appendLog(`后端已连接：${data.service} / Python ${data.python}`);
      }
    } catch (error) {
      setHealth(null);
      if (verbose) {
        appendLog(error instanceof Error ? error.message : "后端连接失败。");
      }
    }
  }

  function parseConfig(): Record<string, unknown> {
    const trimmed = configText.trim();
    if (!trimmed) {
      return {};
    }
    const parsed = JSON.parse(trimmed) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("配置必须是 JSON object。");
    }
    return parsed as Record<string, unknown>;
  }

  async function chooseInput() {
    const selected = await window.railCurve.openPointCloudDialog();
    if (selected) {
      setInputPath(selected);
      appendLog(`选择点云：${selected}`);
    }
  }

  async function chooseInputFolder() {
    const selected = await window.railCurve.openPointCloudFolderDialog();
    if (selected) {
      setInputPath(selected);
      appendLog(`选择 DJI Terra 项目文件夹：${selected}`);
    }
  }

  async function chooseOutput() {
    const selected = await window.railCurve.selectOutputDirectory();
    if (selected) {
      setOutputDir(selected);
      appendLog(`选择输出目录：${selected}`);
    }
  }

  async function chooseDom() {
    const selected = await window.railCurve.openDomDialog();
    if (selected) {
      setDomPath(selected);
      appendLog(`选择 DOM：${selected}`);
    }
  }

  async function chooseModel() {
    const selected = await window.railCurve.openModelDialog();
    if (selected) {
      setModelPath(selected);
      appendLog(`选择语义分割权重：${selected}`);
    }
  }

  async function chooseDsm() {
    const selected = await window.railCurve.openDsmDialog();
    if (selected) {
      setDsmPath(selected);
      appendLog(`选择 DSM：${selected}`);
    }
  }

  async function chooseLasDir() {
    const selected = await window.railCurve.openLasDirectoryDialog();
    if (selected) {
      setLasDir(selected);
      appendLog(`选择 LAS 目录：${selected}`);
    }
  }

  async function loadDefaultConfig() {
    setBusy(true);
    try {
      const config = await apiRequest<Record<string, unknown>>("/api/config/default");
      setConfigText(JSON.stringify(config, null, 2));
      appendLog("已加载默认算法配置。");
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "加载默认配置失败。");
    } finally {
      setBusy(false);
    }
  }

  async function inspectPointCloud() {
    if (!inputPath.trim()) {
      appendLog("请先选择点云文件。");
      return;
    }
    setBusy(true);
    try {
      const info = await apiRequest<PointCloudInfo>("/api/point-cloud/info", {
        method: "POST",
        body: JSON.stringify({ input_path: inputPath })
      });
      setPointInfo(info);
      setActivePage("annotate");
      appendLog(`点云信息：${info.points.toLocaleString()} 点，RGB=${info.has_rgb ? "是" : "否"}。`);
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "读取点云信息失败。");
    } finally {
      setBusy(false);
    }
  }

  async function loadPreview(bounds?: Bounds2d, maxPoints = OVERVIEW_PREVIEW_POINTS) {
    await loadPreviewLayer(bounds ? "focus" : "overview", bounds, maxPoints, true);
  }

  async function loadPreviewLayer(layer: "overview" | "focus", bounds: Bounds2d | undefined, maxPoints: number, resetView: boolean) {
    if (!inputPath.trim()) {
      appendLog("请先选择点云文件。");
      return;
    }
    const requestRef = layer === "overview" ? overviewRequestRef : focusRequestRef;
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    setPreviewLoadingMode(layer);
    if (layer === "overview") {
      setBusy(true);
      focusRequestRef.current += 1;
      setFocusPreviewData(null);
      lastFocusBoundsKeyRef.current = "";
    }
    try {
      const preview = await apiRequest<PreviewData>("/api/point-cloud/preview", {
        method: "POST",
        body: JSON.stringify({ input_path: inputPath, max_points: maxPoints, bounds })
      });
      if (requestRef.current !== requestId) {
        return;
      }
      if (layer === "overview") {
        setPreviewData(preview);
        setPreviewLodMessage(`全图低密度预览：${preview.sample_points.toLocaleString()} / ${preview.input_points.toLocaleString()} 点。`);
        if (resetView) {
          setResetViewToken((value) => value + 1);
        }
      } else {
        setFocusPreviewData(preview);
        setPreviewLodMessage(`当前视口高清：${preview.sample_points.toLocaleString()} / ${preview.input_points.toLocaleString()} 点。`);
      }
      setActivePage("annotate");
      appendLog(
        layer === "focus"
          ? `自动LOD视口高清已加载：显示 ${preview.sample_points.toLocaleString()} / ${preview.input_points.toLocaleString()} 点。`
          : `标注底图已加载：显示 ${preview.sample_points.toLocaleString()} / ${preview.input_points.toLocaleString()} 点。`
      );
    } catch (error) {
      if (requestRef.current === requestId) {
        appendLog(error instanceof Error ? error.message : "加载点云预览失败。");
      }
    } finally {
      if (requestRef.current === requestId) {
        setPreviewLoadingMode("idle");
        if (layer === "overview") {
          setBusy(false);
        }
      }
    }
  }

  async function loadHighDetailPreview() {
    if (!currentViewBounds) {
      appendLog("当前还没有可重载的视图范围。");
      return;
    }
    const fullBounds = pointInfo ? boundsFromPointInfo(pointInfo) : null;
    const paddedBounds = fullBounds ? expandBounds(currentViewBounds, 0.75, fullBounds) : currentViewBounds;
    await loadPreviewLayer("focus", paddedBounds, FOCUSED_PREVIEW_POINTS, false);
  }

  function resetPreviewView() {
    setResetViewToken((value) => value + 1);
  }

  async function openViewer() {
    if (!inputPath.trim()) {
      appendLog("请先选择点云文件。");
      return;
    }
    setBusy(true);
    try {
      const fullBounds = pointInfo ? boundsFromPointInfo(pointInfo) : null;
      const viewerBounds =
        currentViewBounds && fullBounds && boundsArea(currentViewBounds) / Math.max(boundsArea(fullBounds), 1e-6) < 0.92
          ? expandBounds(currentViewBounds, 0.2, fullBounds)
          : null;
      const result = await apiRequest<ApiResult>("/api/viewer/open", {
        method: "POST",
        body: JSON.stringify({
          input_path: inputPath,
          max_points: OPEN3D_VIEWER_MAX_POINTS,
          point_size: OPEN3D_VIEWER_POINT_SIZE,
          bounds: viewerBounds ?? undefined
        })
      });
      setLastResult(result);
      appendLog(
        viewerBounds
          ? `已请求打开 Open3D 当前视口高清窗口：最多 ${OPEN3D_VIEWER_MAX_POINTS.toLocaleString()} 点。`
          : `已请求打开 Open3D 全图采样窗口：最多 ${OPEN3D_VIEWER_MAX_POINTS.toLocaleString()} 点；放大到轨道附近后再点会更清晰。`
      );
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "打开 Open3D 独立窗口失败。");
    } finally {
      setBusy(false);
    }
  }

  async function refreshEmbeddedOpen3DStatus(verbose: boolean) {
    const previousState = open3dStatus?.state;
    try {
      const status = await apiRequest<EmbeddedOpen3DStatus>("/api/viewer/embedded/status");
      setOpen3dStatus(status);
      if (status.pid) {
        setOpen3dViewerPid(status.pid);
      }
      if (status.ready) {
        if (!open3dViewerUrl) {
          setOpen3dViewerUrl(`${status.url}?t=${Date.now()}`);
        }
        if (previousState !== "ready") {
          appendLog(`Open3D 采样画布已就绪：显示 ${formatOptionalNumber(status.display_points ?? status.loaded_points)} 点。`);
        }
      } else if (status.state === "failed" && previousState !== "failed") {
        appendLog(status.error ? `Open3D 加载失败：${status.error}` : status.message || "Open3D 加载失败。");
      } else if (verbose) {
        appendLog(status.message || "Open3D 正在加载。");
      }
    } catch (error) {
      if (verbose) {
        appendLog(error instanceof Error ? error.message : "读取 Open3D 加载状态失败。");
      }
    }
  }

  async function startEmbeddedOpen3D() {
    if (!inputPath.trim()) {
      appendLog("请先选择点云文件。");
      return;
    }
    setBusy(true);
    setOpen3dViewerUrl(null);
    setOpen3dStatus({
      state: "starting",
      phase: "starting_process",
      message: "正在启动 Open3D 子进程",
      percent: null,
      url: "http://127.0.0.1:8888",
      ready: false,
      log_path: "",
      progress_path: ""
    });
    try {
      appendLog(`正在启动内嵌 Open3D 采样查看；最多加载 ${OPEN3D_SAFE_MAX_POINTS.toLocaleString()} 点，避免爆内存。`);
      const result = await apiRequest<EmbeddedOpen3DResult>("/api/viewer/embedded/start", {
        method: "POST",
        body: JSON.stringify({ input_path: inputPath, max_points: OPEN3D_SAFE_MAX_POINTS, point_size: 1 })
      });
      setOpen3dViewerPid(result.pid);
      setOpen3dStatus(result.status ?? {
        state: "starting",
        phase: "starting_process",
        message: "Open3D 子进程已启动，正在等待加载状态",
        percent: null,
        url: result.url,
        pid: result.pid,
        ready: false,
        log_path: result.log_path,
        progress_path: result.progress_path
      });
      setLastResult(result as unknown as ApiResult);
      appendLog(`内嵌 Open3D 子进程已启动：PID ${result.pid}，模式：${result.full_density ? "全量原始点云" : "采样点云"}。`);
      void refreshEmbeddedOpen3DStatus(false);
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "启动内嵌 Open3D 失败。");
      setOpen3dStatus(null);
    } finally {
      setBusy(false);
    }
  }

  async function stopEmbeddedOpen3D() {
    setBusy(true);
    try {
      await apiRequest<ApiResult>("/api/viewer/embedded/stop", {
        method: "POST",
        body: JSON.stringify({})
      });
      setOpen3dViewerUrl(null);
      setOpen3dViewerPid(null);
      setOpen3dStatus(null);
      appendLog("已停止内嵌 Open3D，切回普通标注画布。");
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "停止内嵌 Open3D 失败。");
    } finally {
      setBusy(false);
    }
  }

  function applyRoiToConfig(roi: RoiBox) {
    try {
      const currentConfig = parseConfig();
      const nextConfig: Record<string, unknown> = {
        ...currentConfig,
        roi
      };
      if (roiMode === "auto_tracks") {
        const existingAutoSplit =
          typeof currentConfig.auto_track_split === "object" && currentConfig.auto_track_split !== null
            ? currentConfig.auto_track_split as Record<string, unknown>
            : {};
        nextConfig.auto_track_split = {
          ...existingAutoSplit,
          enabled: true,
          count: autoTrackCount,
          roi
        };
        nextConfig.tracks = [];
      }
      setConfigText(JSON.stringify(nextConfig, null, 2));
      setSelectedRoi(roi);
      appendLog(
        roiMode === "auto_tracks"
          ? `已写入自动多轨 ROI：${autoTrackCount} 条。`
          : "已写入全局 ROI。"
      );
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "ROI 写入配置失败。");
    }
  }

  function addGuidedTrack() {
    const nextId = Math.max(0, ...guidedTracks.map((track) => track.id)) + 1;
    const nextTracks = [...guidedTracks, { id: nextId, points: [], corridor_width: 5.0 }];
    setGuidedTracks(nextTracks);
    setActiveGuidedTarget({ kind: "track", id: nextId });
    syncGuidedPathsToConfig(nextTracks, guidedTurnouts);
    appendLog(`已新增轨道 ${nextId}，请在底图上沿轨道内部点选路径点。`);
  }

  function addGuidedTurnout() {
    const nextId = Math.max(0, ...guidedTurnouts.map((turnout) => turnout.id)) + 1;
    const nextTurnouts = [...guidedTurnouts, { id: nextId, main_points: [], branch_points: [], corridor_width: 7.0 }];
    setGuidedTurnouts(nextTurnouts);
    setActiveGuidedTarget({ kind: "turnout_main", id: nextId });
    syncGuidedPathsToConfig(guidedTracks, nextTurnouts);
    appendLog(`已新增道岔 ${nextId}，先点主线，再切到分支点。`);
  }

  function clearGuidedPaths() {
    const resetTracks = [{ id: 1, points: [], corridor_width: 5.0 }];
    const resetTurnouts: GuidedTurnoutDraft[] = [];
    setGuidedTracks(resetTracks);
    setGuidedTurnouts(resetTurnouts);
    setActiveGuidedTarget({ kind: "track", id: 1 });
    syncGuidedPathsToConfig(resetTracks, resetTurnouts);
    appendLog("已清空人工路径点。");
  }

  function undoGuidedPoint() {
    let changed = false;
    let nextTracks = guidedTracks;
    let nextTurnouts = guidedTurnouts;
    if (activeGuidedTarget.kind === "track") {
      nextTracks = guidedTracks.map((track) => {
        if (track.id !== activeGuidedTarget.id || track.points.length === 0) {
          return track;
        }
        changed = true;
        return { ...track, points: track.points.slice(0, -1) };
      });
      setGuidedTracks(nextTracks);
    } else {
      nextTurnouts = guidedTurnouts.map((turnout) => {
        if (turnout.id !== activeGuidedTarget.id) {
          return turnout;
        }
        if (activeGuidedTarget.kind === "turnout_main" && turnout.main_points.length > 0) {
          changed = true;
          return { ...turnout, main_points: turnout.main_points.slice(0, -1) };
        }
        if (activeGuidedTarget.kind === "turnout_branch" && turnout.branch_points.length > 0) {
          changed = true;
          return { ...turnout, branch_points: turnout.branch_points.slice(0, -1) };
        }
        return turnout;
      });
      setGuidedTurnouts(nextTurnouts);
    }
    if (changed) {
      syncGuidedPathsToConfig(nextTracks, nextTurnouts);
      appendLog("已撤销当前路径的最后一个点。");
    }
  }

  function setActiveGuidedTargetFromKey(key: string) {
    setActiveGuidedTarget(guidedTargetFromKey(key));
  }

  function appendGuidedPoint(point: [number, number]) {
    let nextTracks = guidedTracks;
    let nextTurnouts = guidedTurnouts;
    if (activeGuidedTarget.kind === "track") {
      nextTracks = guidedTracks.map((track) =>
        track.id === activeGuidedTarget.id
          ? { ...track, points: [...track.points, point] }
          : track
      );
      setGuidedTracks(nextTracks);
    } else {
      nextTurnouts = guidedTurnouts.map((turnout) => {
        if (turnout.id !== activeGuidedTarget.id) {
          return turnout;
        }
        if (activeGuidedTarget.kind === "turnout_main") {
          return { ...turnout, main_points: [...turnout.main_points, point] };
        }
        return { ...turnout, branch_points: [...turnout.branch_points, point] };
      });
      setGuidedTurnouts(nextTurnouts);
    }
    syncGuidedPathsToConfig(nextTracks, nextTurnouts);
    appendLog(`已添加路径点：${point[0].toFixed(3)}, ${point[1].toFixed(3)}`);
  }

  function syncGuidedPathsToConfig(nextTracks: GuidedTrackDraft[], nextTurnouts: GuidedTurnoutDraft[]) {
    try {
      const currentConfig = parseConfig();
      const existingGuided =
        typeof currentConfig.guided_paths === "object" && currentConfig.guided_paths !== null
          ? currentConfig.guided_paths as Record<string, unknown>
          : {};
      const validTracks = nextTracks
        .filter((track) => track.points.length >= 2)
        .map((track) => ({
          id: track.id,
          enabled: true,
          points: track.points,
          corridor_width: track.corridor_width
        }));
      const validTurnouts = nextTurnouts
        .filter((turnout) => turnout.main_points.length >= 2 && turnout.branch_points.length >= 2)
        .map((turnout) => ({
          id: turnout.id,
          enabled: true,
          main_points: turnout.main_points,
          branch_points: turnout.branch_points,
          corridor_width: turnout.corridor_width
        }));
      const nextConfig: Record<string, unknown> = {
        ...currentConfig,
        guided_paths: {
          ...existingGuided,
          enabled: validTracks.length > 0 || validTurnouts.length > 0,
          tracks: validTracks,
          turnouts: validTurnouts
        }
      };
      setConfigText(JSON.stringify(nextConfig, null, 2));
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "人工路径点写入配置失败。");
    }
  }

  async function startDomPipeline() {
    if (!canRunDomPipeline) {
      appendLog("请先选择 DOM、DeepLab 权重和输出目录。");
      return;
    }
    setBusy(true);
    try {
      const status = await apiRequest<DomPipelineStatus>("/api/dom-pipeline/start", {
        method: "POST",
        body: JSON.stringify({
          dom_path: domPath,
          model_path: modelPath,
          output_dir: outputDir,
          dsm_path: dsmPath.trim() || null,
          las_dir: lasDir.trim() || null,
          profile: "strict-auto",
          device: pipelineDevice.trim() || "cuda",
          threshold: Number(pipelineThreshold) || 0.5,
          max_tiles: Math.max(0, Number(pipelineMaxTiles) || 0),
          force: pipelineForce,
          epsg: 32651
        })
      });
      setPipelineJobId(status.job_id);
      setPipelineStatus(status);
      setLastResult(compactDomPipelineStatus(status));
      setActivePage("export");
      appendLog(`DOM 流水线已启动：${status.job_id}`);
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "DOM 流水线启动失败。");
    } finally {
      setBusy(false);
    }
  }

  async function refreshDomPipelineStatus(verbose: boolean) {
    if (!pipelineJobId) {
      return;
    }
    const previousState = pipelineStatus?.state;
    try {
      const status = await apiRequest<DomPipelineStatus>(`/api/dom-pipeline/status/${pipelineJobId}`);
      setPipelineStatus(status);
      setLastResult(compactDomPipelineStatus(status));
      if (status.state === "completed" && previousState !== "completed") {
        appendLog(`DOM 流水线完成：${status.outputs?.centerline_3d_shp ?? status.out_dir}`);
      } else if (status.state === "failed" && previousState !== "failed") {
        appendLog(status.error || status.message || "DOM 流水线失败。");
      } else if (verbose) {
        appendLog(status.message || "DOM 流水线状态已刷新。");
      }
    } catch (error) {
      if (verbose) {
        appendLog(error instanceof Error ? error.message : "读取 DOM 流水线状态失败。");
      }
    }
  }

  async function stopDomPipeline() {
    if (!pipelineJobId) {
      return;
    }
    setBusy(true);
    try {
      const status = await apiRequest<DomPipelineStatus>(`/api/dom-pipeline/stop/${pipelineJobId}`, {
        method: "POST"
      });
      setPipelineStatus(status);
      setLastResult(compactDomPipelineStatus(status));
      appendLog("DOM 流水线已停止。");
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "停止 DOM 流水线失败。");
    } finally {
      setBusy(false);
    }
  }

  async function exportCenterline() {
    if (!canRun) {
      appendLog("请先选择点云和输出目录。");
      return;
    }
    setBusy(true);
    try {
      const config = parseConfig();
      appendLog("开始分析并导出中心线，这一步可能需要一些时间。");
      const result = await apiRequest<ApiResult>("/api/export", {
        method: "POST",
        body: JSON.stringify({
          input_path: inputPath,
          output_dir: outputDir,
          config_overrides: config
        })
      });
      const overlay = extractResultOverlay(result);
      setResultOverlay(overlay);
      setLastResult(compactResultForDisplay(result));
      setActivePage("annotate");
      appendLog(
        overlay
          ? `导出完成，结果已叠加到点云画布：${overlay.track_count} 条轨道，${overlay.turnout_count} 个道岔。`
          : "导出完成，但后端没有返回可叠加的结果图层。"
      );
      if (!previewData) {
        appendLog("还没有点云底图；点击“加载全图”后即可看到结果叠加。");
      }
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "分析导出失败。");
    } finally {
      setBusy(false);
    }
  }

  return (
    <FluentProvider theme={webLightTheme}>
      <main className="app-shell-pages">
        <header className="app-commandbar">
          <div className="commandbar-title">
            <Text className="eyebrow">DOM 语义分割流水线</Text>
            <Title2 as="h1">轨道 3D 中心线自动提取</Title2>
            <Text size={200}>主流程：原始 DOM + DeepLab 权重 {"->"} 语义分割 {"->"} 后处理 {"->"} 3D 中心线文件。</Text>
          </div>
          <nav className="page-tabs" aria-label="主流程">
            <PageTab active={activePage === "data"} label="DOM 流水线" onClick={() => setActivePage("data")} />
            <PageTab active={activePage === "export"} label="日志与结果" onClick={() => setActivePage("export")} />
          </nav>
          <div className="topbar-status">
            <Badge appearance="filled" color={health?.ok ? "success" : "warning"}>
              {health?.ok ? `后端在线 · Python ${health.python}` : "后端启动中"}
            </Badge>
            <Text size={200} className="mono-text">{backend?.baseUrl ?? "等待后端地址"}</Text>
            <Button onClick={() => void checkHealth(true)} disabled={busy}>检查后端</Button>
            {busy || pipelineRunning ? <ProgressBar thickness="medium" value={pipelineRunning ? pipelineProgressValue : undefined} /> : null}
          </div>
        </header>

        {activePage === "data" ? (
          <section className="page-shell data-page">
            <Card className="page-card primary-card dom-pipeline-card">
              <CardHeader
                header={<Title3 as="h3">DOM {"->"} 3D 中心线</Title3>}
                description={<Text>这是现在的主流程：输入原始 DOM 和 DeepLab 权重，后端会按 strict-auto 路线完成语义分割、2D 后处理、LAS/DSM 补 Z，并输出 final_delivery。</Text>}
              />
              <div className="dom-pipeline-form">
                <Field label="原始 DOM">
                  <div className="path-field">
                    <Input
                      value={domPath}
                      onChange={(event) => setDomPath(event.currentTarget.value)}
                      placeholder="例如 D:/data/dom.tif"
                      appearance="outline"
                    />
                    <Button onClick={() => void chooseDom()} disabled={busy || pipelineRunning}>选择 DOM</Button>
                  </div>
                </Field>
                <Field label="DeepLab 权重">
                  <div className="path-field">
                    <Input
                      value={modelPath}
                      onChange={(event) => setModelPath(event.currentTarget.value)}
                      placeholder="rail_semantic_deeplab_resnet50.pt"
                      appearance="outline"
                    />
                    <Button onClick={() => void chooseModel()} disabled={busy || pipelineRunning}>选择权重</Button>
                  </div>
                </Field>
                <Field label="输出目录">
                  <div className="path-field">
                    <Input
                      value={outputDir}
                      onChange={(event) => setOutputDir(event.currentTarget.value)}
                      placeholder="例如 D:/rail-curve-extractor/output/dom_centerline_strict_auto_v1"
                      appearance="outline"
                    />
                    <Button onClick={() => void chooseOutput()} disabled={busy || pipelineRunning}>选择目录</Button>
                  </div>
                </Field>
                <details className="config-details">
                  <summary>DSM / LAS 与运行参数</summary>
                  <div className="dom-grid-two">
                    <Field label="DSM 栅格">
                      <div className="path-field">
                        <Input value={dsmPath} onChange={(event) => setDsmPath(event.currentTarget.value)} placeholder="可留空使用脚本默认值" appearance="outline" />
                        <Button onClick={() => void chooseDsm()} disabled={busy || pipelineRunning}>选择 DSM</Button>
                      </div>
                    </Field>
                    <Field label="LAS/LAZ 目录">
                      <div className="path-field">
                        <Input value={lasDir} onChange={(event) => setLasDir(event.currentTarget.value)} placeholder="可留空使用脚本默认值" appearance="outline" />
                        <Button onClick={() => void chooseLasDir()} disabled={busy || pipelineRunning}>选择目录</Button>
                      </div>
                    </Field>
                    <Field label="推理设备">
                      <Input value={pipelineDevice} onChange={(event) => setPipelineDevice(event.currentTarget.value)} placeholder="cuda / cpu" appearance="outline" />
                    </Field>
                    <Field label="分割阈值">
                      <Input value={pipelineThreshold} onChange={(event) => setPipelineThreshold(event.currentTarget.value)} type="number" min={0} max={1} step={0.01} appearance="outline" />
                    </Field>
                    <Field label="最大瓦片数">
                      <Input value={pipelineMaxTiles} onChange={(event) => setPipelineMaxTiles(event.currentTarget.value)} type="number" min={0} appearance="outline" />
                    </Field>
                    <label className="radio-pill force-toggle">
                      <input type="checkbox" checked={pipelineForce} onChange={(event) => setPipelineForce(event.currentTarget.checked)} />
                      强制重跑已有阶段
                    </label>
                  </div>
                </details>
                <DomPipelineProgress status={pipelineStatus} progressValue={pipelineProgressValue} />
                <div className="button-row">
                  <Button appearance="primary" size="large" onClick={() => void startDomPipeline()} disabled={!canRunDomPipeline}>
                    {busy ? <Spinner size="tiny" /> : null}
                    开始 DOM 流水线
                  </Button>
                  <Button onClick={() => void refreshDomPipelineStatus(true)} disabled={!pipelineJobId || busy}>刷新进度</Button>
                  <Button onClick={() => void stopDomPipeline()} disabled={!pipelineRunning || busy}>停止</Button>
                </div>
              </div>
            </Card>

            <Card className="page-card primary-card legacy-pointcloud-card">
              <CardHeader
                header={<Title3 as="h3">选择数据源</Title3>}
                description={<Text>可以选单个 LAS/LAZ，也可以直接选整个 DJI Terra 输出目录，软件会自动找分块点云。</Text>}
              />
              <Field label="点云文件 / DJI Terra 文件夹">
                <div className="path-field">
                  <Input
                    value={inputPath}
                    onChange={(event) => setInputPath(event.currentTarget.value)}
                    placeholder="选择 .las/.laz，或粘贴 D:\\nantong_port_las"
                    appearance="outline"
                  />
                </div>
              </Field>
              <div className="button-row">
                <Button onClick={() => void chooseInput()} disabled={busy}>选择文件</Button>
                <Button onClick={() => void chooseInputFolder()} disabled={busy}>选择 DJI 文件夹</Button>
                <Button appearance="primary" onClick={() => void inspectPointCloud()} disabled={busy || !inputPath}>
                  {busy ? <Spinner size="tiny" /> : null}
                  读取点云信息
                </Button>
              </div>
            </Card>

            <Card className="page-card legacy-pointcloud-card">
              <CardHeader header={<Title3 as="h3">点云概览</Title3>} />
              {pointInfo ? (
                <>
                  <div className="metric-grid">
                    <Metric label="点数" value={pointInfo.points.toLocaleString()} />
                    <Metric label="分块" value={String(pointInfo.source_file_count ?? 1)} />
                    <Metric label="RGB" value={pointInfo.has_rgb ? "有" : "无"} />
                    <Metric label="强度" value={pointInfo.has_intensity ? "有" : "无"} />
                    <Metric label="格式" value={pointInfo.source_format || "--"} />
                  </div>
                  <Divider />
                  <div className="bounds-box">
                    <Text weight="semibold">坐标范围</Text>
                    <Text size={200}>Min: {formatVector(pointInfo.bounds.minimum)}</Text>
                    <Text size={200}>Max: {formatVector(pointInfo.bounds.maximum)}</Text>
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <Text size={600}>还没有点云概览</Text>
                  <Text>先选路径并读取信息；成功后会自动进入“标注工作台”。</Text>
                </div>
              )}
            </Card>

            <Card className="page-card guide-card legacy-pointcloud-card">
              <CardHeader header={<Title3 as="h3">下一步</Title3>} />
              <Text>1. 读取点云信息，确认 RGB、点数和分块。</Text>
              <Text>2. 进入标注工作台，加载全图后放大到轨道附近。</Text>
              <Text>3. 放大到轨道附近后软件会自动补当前视口高清点云，再框选区域或点人工路径点。</Text>
              <Button appearance="primary" onClick={() => setActivePage("annotate")} disabled={!inputPath}>
                去标注工作台
              </Button>
            </Card>
          </section>
        ) : null}

        {activePage === "annotate" ? (
          <section className="annotate-shell">
            <aside className="annotation-tools">
              <Card className="tool-card">
                <CardHeader
                  header={<Title3 as="h3">画布</Title3>}
                  description={<Text>默认使用自动LOD标注画布：全图低密度，缩放后自动加载视口高清点云。</Text>}
                />
                <div className="button-row vertical-buttons">
                  <Button appearance="primary" onClick={() => void loadPreview()} disabled={busy || previewLoadingMode !== "idle" || !inputPath}>
                    {previewLoadingMode === "overview" ? <Spinner size="tiny" /> : null}
                    自动LOD标注画布
                  </Button>
                  <Button onClick={() => void openViewer()} disabled={busy || !inputPath}>Open3D当前视口高清</Button>
                  <Button onClick={resetPreviewView} disabled={!previewData}>重置视图</Button>
                  <Button onClick={() => void stopEmbeddedOpen3D()} disabled={busy || (!open3dViewerUrl && !open3dStatus)}>切回标注画布</Button>
                </div>
                <Text size={200} className="muted-text">
                  {previewLodMessage} Open3D 会优先读取当前画布视口，最多 1200 万点；标注仍用自动LOD画布。
                </Text>
              </Card>

              <Card className="tool-card">
                <CardHeader
                  header={<Title3 as="h3">区域</Title3>}
                  description={<Text>框住需要计算的轨道范围。</Text>}
                />
                <div className="segmented-block">
                  <label className="radio-pill">
                    <input type="radio" checked={roiMode === "global"} onChange={() => setRoiMode("global")} />
                    单区域
                  </label>
                  <label className="radio-pill">
                    <input type="radio" checked={roiMode === "auto_tracks"} onChange={() => setRoiMode("auto_tracks")} />
                    框内多轨
                  </label>
                </div>
                <div className="form-line">
                  <Field label="框内轨道数">
                    <Input
                      type="number"
                      min={1}
                      max={12}
                      value={String(autoTrackCount)}
                      onChange={(event) => setAutoTrackCount(Math.max(1, Math.min(12, Number(event.currentTarget.value) || 1)))}
                      disabled={roiMode !== "auto_tracks"}
                    />
                  </Field>
                  <Badge appearance={selectedRoi ? "filled" : "outline"} color={selectedRoi ? "success" : "informative"}>
                    {selectedRoi ? "已框选" : "未框选"}
                  </Badge>
                </div>
              </Card>

              <Card className="tool-card">
                <CardHeader
                  header={<Title3 as="h3">路径点</Title3>}
                  description={<Text>点不必在中心，但要在轨道内部。</Text>}
                />
                <label className="radio-pill full-width-pill anchor-toggle">
                  <input
                    type="checkbox"
                    checked={guidedPickEnabled}
                    onChange={(event) => setGuidedPickEnabled(event.currentTarget.checked)}
                  />
                  开启点选路径
                </label>
                <select
                  className="target-select wide"
                  value={guidedTargetToKey(activeGuidedTarget)}
                  onChange={(event) => setActiveGuidedTargetFromKey(event.currentTarget.value)}
                >
                  {guidedTargetOptions.map((option) => (
                    <option key={option.key} value={option.key}>{option.label}</option>
                  ))}
                </select>
                <div className="button-row">
                  <Button onClick={addGuidedTrack} disabled={busy}>新增轨道</Button>
                  <Button onClick={addGuidedTurnout} disabled={busy}>新增道岔</Button>
                  <Button onClick={undoGuidedPoint} disabled={busy}>撤销点</Button>
                  <Button onClick={clearGuidedPaths} disabled={busy}>清空</Button>
                </div>
                <Text size={200} className="muted-text">普通轨道点 2–6 个；道岔分别点主线和分支。</Text>
              </Card>

              <Card className="tool-card">
                <CardHeader
                  header={<Title3 as="h3">结果图层</Title3>}
                  description={<Text>算法跑完后会叠加到点云上，用来人工判断是否贴轨。</Text>}
                />
                {resultOverlay ? (
                  <>
                    <div className="metric-grid compact-metrics">
                      <Metric label="轨道" value={String(resultOverlay.track_count)} />
                      <Metric label="道岔" value={String(resultOverlay.turnout_count)} />
                    </div>
                    <div className="result-layer-list">
                      {resultOverlay.tracks.slice(0, 6).map((track) => (
                        <Text key={track.id} size={200}>
                          {track.label} · 置信度 {(track.confidence * 100).toFixed(0)}%
                        </Text>
                      ))}
                    </div>
                    <Button onClick={() => setResultOverlay(null)} disabled={busy}>隐藏结果图层</Button>
                  </>
                ) : (
                  <Text size={200} className="muted-text">暂无结果。点击“分析并导出”后会显示中心线、钢轨候选点和道岔点。</Text>
                )}
              </Card>
            </aside>

            <section className="canvas-workspace">
              <header className="canvas-toolbar">
                <div className="canvas-title">
                  <Text className="eyebrow">点云画布</Text>
                  <Title2 as="h2">{open3dViewerUrl ? "Open3D 采样画布" : open3dStatus ? "Open3D 加载中" : "自动LOD标注画布"}</Title2>
                  <Text size={200}>
                    {open3dViewerUrl
                      ? "当前显示 Open3D WebRTC 采样渲染流；超大点云的精细标注建议切回自动LOD画布。"
                      : open3dStatus
                        ? "正在按真实文件和点数加载；没有真实分母时只显示阶段，不伪造百分比。"
                      : "滚轮缩放，右键/中键平移；停下后会自动加载当前视口高清点云。"}
                  </Text>
                </div>
                <Button appearance="primary" onClick={() => setActivePage("export")}>去导出</Button>
              </header>
              <div className="viewer-canvas-card">
                {open3dViewerUrl ? (
                  <iframe
                    className="open3d-iframe"
                    title="Open3D embedded point cloud viewer"
                    src={open3dViewerUrl}
                    allow="autoplay; fullscreen"
                  />
                ) : open3dStatus ? (
                  <Open3DProgressPanel
                    status={open3dStatus}
                    onStop={() => void stopEmbeddedOpen3D()}
                  />
                ) : (
                  <RoiCanvas
                    preview={previewData}
                    focusPreview={focusPreviewData}
                    focusLoading={previewLoadingMode === "focus"}
                    selectedRoi={selectedRoi}
                    onSelect={applyRoiToConfig}
                    guidedPickEnabled={guidedPickEnabled}
                    guidedTracks={guidedTracks}
                    guidedTurnouts={guidedTurnouts}
                    resultOverlay={resultOverlay}
                    activeGuidedTarget={activeGuidedTarget}
                    onGuidedPoint={appendGuidedPoint}
                    onViewBoundsChange={setCurrentViewBounds}
                    resetViewToken={resetViewToken}
                  />
                )}
              </div>
              <footer className="viewer-statusbar">
                <Text size={200}>
                  {open3dViewerUrl
                    ? `Open3D 采样点云${open3dViewerPid ? ` · PID ${open3dViewerPid}` : ""}`
                    : open3dStatus
                      ? `Open3D ${open3dStatus.state} · ${open3dStatus.message || open3dStatus.phase}`
                    : previewData
                      ? `全图 ${previewData.sample_points.toLocaleString()} 点${focusPreviewData ? ` · 视口高清 ${focusPreviewData.sample_points.toLocaleString()} 点` : ""}${previewLoadingMode === "focus" ? " · 正在自动补细节" : ""}`
                      : "尚未加载点云底图"}
                </Text>
                <Text size={200}>
                  {selectedRoi
                    ? `区域 X ${selectedRoi.x_min.toFixed(2)}–${selectedRoi.x_max.toFixed(2)} / Y ${selectedRoi.y_min.toFixed(2)}–${selectedRoi.y_max.toFixed(2)}`
                    : "未框选区域"}
                </Text>
                <Text size={200}>
                  人工点：轨道 {guidedTracks.reduce((sum, track) => sum + track.points.length, 0)}，
                  道岔 {guidedTurnouts.reduce((sum, turnout) => sum + turnout.main_points.length + turnout.branch_points.length, 0)}
                </Text>
                <Text size={200}>
                  结果图层：{resultOverlay ? `${resultOverlay.track_count} 条轨道 / ${resultOverlay.turnout_count} 个道岔` : "未生成"}
                </Text>
              </footer>
            </section>
          </section>
        ) : null}

        {activePage === "export" ? (
          <section className="page-shell export-page">
            <Card className="page-card primary-card">
              <CardHeader
                header={<Title3 as="h3">DOM 流水线状态</Title3>}
                description={<Text>这里显示语义分割和后处理进度。完成后重点验收 final_delivery 里的 2D/3D Shapefile。</Text>}
              />
              <DomPipelineProgress status={pipelineStatus} progressValue={pipelineProgressValue} />
              <div className="button-row">
                <Button appearance="primary" size="large" onClick={() => void startDomPipeline()} disabled={!canRunDomPipeline}>
                  {busy ? <Spinner size="tiny" /> : null}
                  开始 DOM 流水线
                </Button>
                <Button onClick={() => void refreshDomPipelineStatus(true)} disabled={!pipelineJobId || busy}>刷新进度</Button>
                <Button onClick={() => setActivePage("data")}>修改输入</Button>
              </div>
              <details className="config-details">
                <summary>旧 JSON 配置（点云中心线已停用）</summary>
                <Button onClick={() => void loadDefaultConfig()} disabled={busy}>加载默认配置</Button>
                <Textarea
                  className="config-editor"
                  value={configText}
                  onChange={(event) => setConfigText(event.currentTarget.value)}
                  resize="vertical"
                  spellCheck={false}
                />
              </details>
            </Card>

            <Card className="page-card">
              <CardHeader header={<Title3 as="h3">运行日志</Title3>} />
              <div className="log-list export-log">
                {logs.map((item) => <div key={item}>{item}</div>)}
              </div>
            </Card>

            <Card className="page-card">
              <CardHeader header={<Title3 as="h3">最近结果</Title3>} />
              <pre className="result-pre">{lastResult ? JSON.stringify(lastResult, null, 2) : "暂无结果"}</pre>
            </Card>
          </section>
        ) : null}
      </main>
    </FluentProvider>
  );
}

function PageTab(props: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button className={`page-tab ${props.active ? "active" : ""}`} type="button" onClick={props.onClick}>
      {props.label}
    </button>
  );
}

function RoiCanvas(props: {
  preview: PreviewData | null;
  focusPreview: PreviewData | null;
  focusLoading: boolean;
  selectedRoi: RoiBox | null;
  onSelect: (roi: RoiBox) => void;
  guidedPickEnabled: boolean;
  guidedTracks: GuidedTrackDraft[];
  guidedTurnouts: GuidedTurnoutDraft[];
  resultOverlay: ResultOverlay | null;
  activeGuidedTarget: GuidedTarget;
  onGuidedPoint: (point: [number, number]) => void;
  onViewBoundsChange: (bounds: Bounds2d | null) => void;
  resetViewToken: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);
  const [panStart, setPanStart] = useState<{ point: { x: number; y: number }; bounds: Bounds2d } | null>(null);
  const [viewBounds, setViewBounds] = useState<Bounds2d | null>(null);

  useEffect(() => {
    draw();
  }, [props.preview, props.focusPreview, props.focusLoading, props.selectedRoi, props.resultOverlay, dragStart, dragCurrent, panStart, viewBounds, props.guidedTracks, props.guidedTurnouts, props.activeGuidedTarget]);

  useEffect(() => {
    const preview = props.preview;
    if (!preview) {
      setViewBounds(null);
      props.onViewBoundsChange(null);
      return;
    }
    setViewBounds((currentBounds) => {
      if (currentBounds) {
        return currentBounds;
      }
      const nextBounds = boundsFromPreview(preview);
      props.onViewBoundsChange(nextBounds);
      return nextBounds;
    });
  }, [props.preview?.input_path]);

  useEffect(() => {
    if (!props.preview) {
      return;
    }
    const nextBounds = boundsFromPreview(props.preview);
    setViewBounds(nextBounds);
    props.onViewBoundsChange(nextBounds);
  }, [props.resetViewToken]);

  useEffect(() => {
    props.onViewBoundsChange(viewBounds);
  }, [viewBounds]);

  function viewMetrics() {
    const canvas = canvasRef.current;
    const preview = props.preview;
    if (!canvas || !preview) {
      return null;
    }
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const padding = 18;
    const activeBounds = viewBounds ?? boundsFromPreview(preview);
    const spanX = Math.max(activeBounds.x_max - activeBounds.x_min, 1e-6);
    const spanY = Math.max(activeBounds.y_max - activeBounds.y_min, 1e-6);
    const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
    const drawWidth = spanX * scale;
    const drawHeight = spanY * scale;
    const offsetX = padding + (width - padding * 2 - drawWidth) / 2;
    const offsetY = padding + (height - padding * 2 - drawHeight) / 2;
    return { width, height, padding, scale, offsetX, offsetY, minX: activeBounds.x_min, minY: activeBounds.y_min, bounds: activeBounds };
  }

  function worldToCanvas(point: [number, number]) {
    const metrics = viewMetrics();
    if (!metrics) {
      return { x: 0, y: 0 };
    }
    return {
      x: metrics.offsetX + (point[0] - metrics.minX) * metrics.scale,
      y: metrics.height - (metrics.offsetY + (point[1] - metrics.minY) * metrics.scale)
    };
  }

  function canvasToWorld(point: { x: number; y: number }): [number, number] {
    const metrics = viewMetrics();
    if (!metrics) {
      return [0, 0] as [number, number];
    }
    return [
      metrics.minX + (point.x - metrics.offsetX) / metrics.scale,
      metrics.minY + (metrics.height - point.y - metrics.offsetY) / metrics.scale
    ] as [number, number];
  }

  function pointerPoint(event: React.PointerEvent<HTMLCanvasElement> | React.WheelEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.max(1, Math.floor(height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#f8fbfd";
    context.fillRect(0, 0, width, height);

    if (!props.preview) {
      context.fillStyle = "#64748b";
      context.font = "14px Microsoft YaHei UI";
      context.textAlign = "center";
      context.fillText("点击“加载全图”后在这里缩放、平移、框选区域", width / 2, height / 2);
      return;
    }

    drawPreviewPoints(context, props.preview, false);
    if (props.focusPreview) {
      drawPreviewPoints(context, props.focusPreview, true);
    }

    if (props.selectedRoi) {
      drawWorldRoi(context, props.selectedRoi, "#16a34a");
    }
    drawResultOverlay(context);
    drawGuidedPaths(context);
    if (dragStart && dragCurrent) {
      drawScreenRect(context, dragStart, dragCurrent, "#147d8f");
    }
    drawViewportHint(context, width, height);
  }

  function drawPreviewPoints(context: CanvasRenderingContext2D, preview: PreviewData, emphasized: boolean) {
    const metrics = viewMetrics();
    const pointSize = emphasized ? Math.max(1.8, previewPointSize() + 0.4) : previewPointSize();
    const alpha = emphasized ? 0.82 : 0.42;
    const fallback = emphasized ? `rgba(15, 118, 110, ${alpha})` : `rgba(100, 116, 139, ${alpha})`;
    for (let index = 0; index < preview.points_xy.length; index += 1) {
      const point = preview.points_xy[index];
      if (metrics && !pointInsideBounds(point, metrics.bounds)) {
        continue;
      }
      const canvasPoint = worldToCanvas(point);
      const color = preview.rgb?.[index];
      context.fillStyle = color ? `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})` : fallback;
      context.fillRect(canvasPoint.x, canvasPoint.y, pointSize, pointSize);
    }
  }

  function previewPointSize() {
    const preview = props.preview;
    const activeBounds = viewBounds;
    if (!preview || !activeBounds) {
      return 1.5;
    }
    const fullSpanX = Math.max(preview.bounds.maximum[0] - preview.bounds.minimum[0], 1e-6);
    const fullSpanY = Math.max(preview.bounds.maximum[1] - preview.bounds.minimum[1], 1e-6);
    const viewSpanX = Math.max(activeBounds.x_max - activeBounds.x_min, 1e-6);
    const viewSpanY = Math.max(activeBounds.y_max - activeBounds.y_min, 1e-6);
    const zoomRatio = Math.sqrt((fullSpanX * fullSpanY) / Math.max(viewSpanX * viewSpanY, 1e-6));
    if (zoomRatio > 16) {
      return 2.8;
    }
    if (zoomRatio > 5) {
      return 2.2;
    }
    return 1.5;
  }

  function drawViewportHint(context: CanvasRenderingContext2D, width: number, height: number) {
    context.fillStyle = "rgba(15, 23, 42, 0.62)";
    context.font = "12px Microsoft YaHei UI";
    context.textAlign = "left";
    context.fillText(
      props.focusLoading
        ? "正在自动加载当前视口高清点云..."
        : "滚轮缩放；右键/中键拖拽平移；停下后自动加载当前视口高清点云",
      18,
      height - 18
    );
  }

  function drawResultOverlay(context: CanvasRenderingContext2D) {
    const overlay = props.resultOverlay;
    if (!overlay) {
      return;
    }
    const palette = ["#ef4444", "#14b8a6", "#f97316", "#8b5cf6", "#22c55e", "#ec4899", "#0ea5e9"];
    for (let index = 0; index < overlay.tracks.length; index += 1) {
      const track = overlay.tracks[index];
      const color = palette[index % palette.length];
      drawResultPoints(context, track.rail_points_xy, color);
    }
    for (let index = 0; index < overlay.tracks.length; index += 1) {
      const track = overlay.tracks[index];
      const color = palette[index % palette.length];
      drawResultPolyline(context, track.centerline_xy, color, track.label);
    }
    for (const turnout of overlay.turnouts) {
      const switchPoint = turnout.switch_point_xy[0];
      if (switchPoint) {
        drawSwitchPoint(context, switchPoint, turnout.label);
      }
    }
  }

  function drawResultPoints(context: CanvasRenderingContext2D, points: [number, number][], color: string) {
    if (points.length === 0) {
      return;
    }
    const metrics = viewMetrics();
    const size = Math.max(2.2, previewPointSize() + 0.8);
    context.fillStyle = hexToRgba(color, 0.42);
    for (const point of points) {
      if (metrics && !pointInsideBounds(point, metrics.bounds)) {
        continue;
      }
      const canvasPoint = worldToCanvas(point);
      context.fillRect(canvasPoint.x - size / 2, canvasPoint.y - size / 2, size, size);
    }
  }

  function drawResultPolyline(
    context: CanvasRenderingContext2D,
    points: [number, number][],
    color: string,
    label: string
  ) {
    if (points.length < 2) {
      return;
    }
    const canvasPoints = points.map((point) => worldToCanvas(point));
    context.setLineDash([]);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "rgba(255, 255, 255, 0.92)";
    context.lineWidth = 7;
    context.beginPath();
    context.moveTo(canvasPoints[0].x, canvasPoints[0].y);
    for (const point of canvasPoints.slice(1)) {
      context.lineTo(point.x, point.y);
    }
    context.stroke();
    context.strokeStyle = color;
    context.lineWidth = 3.2;
    context.beginPath();
    context.moveTo(canvasPoints[0].x, canvasPoints[0].y);
    for (const point of canvasPoints.slice(1)) {
      context.lineTo(point.x, point.y);
    }
    context.stroke();

    const labelPoint = canvasPoints[Math.floor(canvasPoints.length / 2)];
    context.fillStyle = hexToRgba(color, 0.92);
    context.font = "12px Microsoft YaHei UI";
    context.textAlign = "left";
    context.fillText(label, labelPoint.x + 8, labelPoint.y - 8);
  }

  function drawSwitchPoint(context: CanvasRenderingContext2D, point: [number, number], label: string) {
    const canvasPoint = worldToCanvas(point);
    context.fillStyle = "#ffffff";
    context.strokeStyle = "#dc2626";
    context.lineWidth = 2.5;
    context.beginPath();
    context.moveTo(canvasPoint.x, canvasPoint.y - 8);
    context.lineTo(canvasPoint.x + 8, canvasPoint.y);
    context.lineTo(canvasPoint.x, canvasPoint.y + 8);
    context.lineTo(canvasPoint.x - 8, canvasPoint.y);
    context.closePath();
    context.fill();
    context.stroke();
    context.fillStyle = "#dc2626";
    context.font = "12px Microsoft YaHei UI";
    context.textAlign = "left";
    context.fillText(label, canvasPoint.x + 10, canvasPoint.y - 10);
  }

  function hexToRgba(hex: string, alpha: number) {
    const normalized = hex.replace("#", "");
    const value = Number.parseInt(normalized, 16);
    const red = (value >> 16) & 255;
    const green = (value >> 8) & 255;
    const blue = value & 255;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function drawGuidedPaths(context: CanvasRenderingContext2D) {
    for (const track of props.guidedTracks) {
      drawGuidedPointSeries(
        context,
        track.points,
        guidedTargetToKey({ kind: "track", id: track.id }) === guidedTargetToKey(props.activeGuidedTarget)
          ? "#2563eb"
          : "#60a5fa",
        `T${track.id}`
      );
    }
    for (const turnout of props.guidedTurnouts) {
      drawGuidedPointSeries(
        context,
        turnout.main_points,
        guidedTargetToKey({ kind: "turnout_main", id: turnout.id }) === guidedTargetToKey(props.activeGuidedTarget)
          ? "#7c3aed"
          : "#a78bfa",
        `M${turnout.id}`
      );
      drawGuidedPointSeries(
        context,
        turnout.branch_points,
        guidedTargetToKey({ kind: "turnout_branch", id: turnout.id }) === guidedTargetToKey(props.activeGuidedTarget)
          ? "#ea580c"
          : "#fb923c",
        `B${turnout.id}`
      );
    }
  }

  function drawGuidedPointSeries(
    context: CanvasRenderingContext2D,
    points: [number, number][],
    color: string,
    label: string
  ) {
    if (points.length === 0) {
      return;
    }
    const canvasPoints = points.map((point) => worldToCanvas(point));
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.setLineDash([]);
    if (canvasPoints.length >= 2) {
      context.beginPath();
      context.moveTo(canvasPoints[0].x, canvasPoints[0].y);
      for (const point of canvasPoints.slice(1)) {
        context.lineTo(point.x, point.y);
      }
      context.stroke();
    }
    for (let index = 0; index < canvasPoints.length; index += 1) {
      const point = canvasPoints[index];
      context.fillStyle = "#ffffff";
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      context.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.fillStyle = color;
      context.font = "11px Microsoft YaHei UI";
      context.textAlign = "left";
      context.fillText(`${label}.${index + 1}`, point.x + 7, point.y - 7);
    }
  }

  function drawWorldRoi(context: CanvasRenderingContext2D, roi: RoiBox, color: string) {
    const topLeft = worldToCanvas([roi.x_min, roi.y_max]);
    const bottomRight = worldToCanvas([roi.x_max, roi.y_min]);
    drawScreenRect(context, topLeft, bottomRight, color);
  }

  function drawScreenRect(context: CanvasRenderingContext2D, start: { x: number; y: number }, end: { x: number; y: number }, color: string) {
    const x = Math.min(start.x, end.x);
    const y = Math.min(start.y, end.y);
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.setLineDash([7, 5]);
    context.strokeRect(x, y, width, height);
    context.setLineDash([]);
    context.fillStyle = color === "#16a34a" ? "rgba(22, 163, 74, 0.10)" : "rgba(20, 125, 143, 0.12)";
    context.fillRect(x, y, width, height);
  }

  function zoomView(screenPoint: { x: number; y: number }, zoomIn: boolean) {
    const metrics = viewMetrics();
    if (!metrics) {
      return;
    }
    const worldPoint = canvasToWorld(screenPoint);
    const factor = zoomIn ? 0.72 : 1.38;
    const current = metrics.bounds;
    const nextBounds: Bounds2d = {
      x_min: worldPoint[0] - (worldPoint[0] - current.x_min) * factor,
      x_max: worldPoint[0] + (current.x_max - worldPoint[0]) * factor,
      y_min: worldPoint[1] - (worldPoint[1] - current.y_min) * factor,
      y_max: worldPoint[1] + (current.y_max - worldPoint[1]) * factor
    };
    setViewBounds(nextBounds);
  }

  function panView(currentPoint: { x: number; y: number }) {
    const metrics = viewMetrics();
    if (!metrics || !panStart) {
      return;
    }
    const deltaX = currentPoint.x - panStart.point.x;
    const deltaY = currentPoint.y - panStart.point.y;
    const shiftX = -deltaX / metrics.scale;
    const shiftY = deltaY / metrics.scale;
    setViewBounds({
      x_min: panStart.bounds.x_min + shiftX,
      x_max: panStart.bounds.x_max + shiftX,
      y_min: panStart.bounds.y_min + shiftY,
      y_max: panStart.bounds.y_max + shiftY
    });
  }

  return (
    <canvas
      ref={canvasRef}
      className="roi-canvas"
      onContextMenu={(event) => event.preventDefault()}
      onWheel={(event) => {
        if (!props.preview) {
          return;
        }
        event.preventDefault();
        zoomView(pointerPoint(event), event.deltaY < 0);
      }}
      onPointerDown={(event) => {
        if (!props.preview) {
          return;
        }
        const point = pointerPoint(event);
        if (event.button === 1 || event.button === 2 || event.shiftKey) {
          const metrics = viewMetrics();
          if (metrics) {
            setPanStart({ point, bounds: metrics.bounds });
            event.currentTarget.setPointerCapture(event.pointerId);
          }
          return;
        }
        if (props.guidedPickEnabled) {
          props.onGuidedPoint(canvasToWorld(point));
          return;
        }
        setDragStart(point);
        setDragCurrent(point);
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (panStart) {
          panView(pointerPoint(event));
          return;
        }
        if (!dragStart) {
          return;
        }
        setDragCurrent(pointerPoint(event));
      }}
      onPointerUp={(event) => {
        if (panStart) {
          setPanStart(null);
          return;
        }
        if (!dragStart || !dragCurrent) {
          setDragStart(null);
          setDragCurrent(null);
          return;
        }
        const first = canvasToWorld(dragStart);
        const second = canvasToWorld(pointerPoint(event));
        const roi: RoiBox = {
          x_min: Math.min(first[0], second[0]),
          x_max: Math.max(first[0], second[0]),
          y_min: Math.min(first[1], second[1]),
          y_max: Math.max(first[1], second[1]),
          z_min: null,
          z_max: null
        };
        if (Math.abs(roi.x_max - roi.x_min) > 0.05 && Math.abs(roi.y_max - roi.y_min) > 0.05) {
          props.onSelect(roi);
        }
        setDragStart(null);
        setDragCurrent(null);
      }}
    />
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric">
      <Text size={200}>{props.label}</Text>
      <Text size={600} weight="semibold">{props.value}</Text>
    </div>
  );
}

function DomPipelineProgress(props: { status: DomPipelineStatus | null; progressValue?: number }) {
  const status = props.status;
  if (!status) {
    return (
      <div className="pipeline-progress idle">
        <Text weight="semibold">尚未运行</Text>
        <Text size={200} className="muted-text">选择 DOM、权重和输出目录后启动。输出会写入 final_delivery。</Text>
      </div>
    );
  }
  const stageText =
    status.stage_index && status.stage_count
      ? `${status.stage_index}/${status.stage_count} ${status.stage_name ?? ""}`
      : status.stage_name ?? "--";
  const percentText =
    typeof status.percent === "number" && Number.isFinite(status.percent)
      ? `${status.percent.toFixed(1)}%`
      : "--";
  const badgeColor = status.state === "failed" ? "danger" : status.state === "completed" ? "success" : "informative";
  return (
    <div className="pipeline-progress">
      <div className="pipeline-progress-head">
        <Badge appearance="filled" color={badgeColor}>{status.state}</Badge>
        <Text weight="semibold">{status.message || status.stage_description || "DOM 流水线运行中"}</Text>
        <Text size={200} className="mono-text">{percentText}</Text>
      </div>
      <ProgressBar className="truth-progress" thickness="large" value={props.progressValue} />
      <div className="pipeline-progress-meta">
        <Metric label="阶段" value={stageText} />
        <Metric label="进程" value={status.pid ? String(status.pid) : "--"} />
        <Metric label="3D 输出" value={String(status.outputs?.centerline_3d_shp ?? "--")} />
        <Metric label="日志" value={lastPathPart(status.log_path)} />
      </div>
      {status.error ? <Text className="error-text">{status.error}</Text> : null}
      <Text size={200} className="mono-text">输出目录：{status.out_dir}</Text>
    </div>
  );
}

function Open3DProgressPanel(props: { status: EmbeddedOpen3DStatus; onStop: () => void }) {
  const { status } = props;
  const hasRealPercent = typeof status.percent === "number" && Number.isFinite(status.percent);
  const progressValue = hasRealPercent ? Math.max(0, Math.min(1, Number(status.percent) / 100)) : undefined;
  const fileText =
    status.file_index && status.file_count
      ? `${status.file_index}/${status.file_count}${status.current_file ? ` · ${lastPathPart(status.current_file)}` : ""}`
      : status.current_file
        ? lastPathPart(status.current_file)
        : "--";
  const pointsText =
    status.loaded_points != null && status.total_points != null
      ? `${formatOptionalNumber(status.loaded_points)} / ${formatOptionalNumber(status.total_points)}`
      : status.loaded_points != null
        ? formatOptionalNumber(status.loaded_points)
        : "--";

  return (
    <div className="open3d-progress-panel">
      <div className="open3d-progress-card">
        <Badge appearance="filled" color={status.state === "failed" ? "danger" : status.ready ? "success" : "informative"}>
          {status.state === "failed" ? "加载失败" : status.ready ? "已就绪" : "真实加载中"}
        </Badge>
        <Title2 as="h2">{status.message || "正在加载 Open3D 采样点云"}</Title2>
        <Text className="muted-text">
          {hasRealPercent
            ? `进度来自已读取点数 / 总点数：${status.percent?.toFixed(2)}%`
            : "还没有可计算的真实分母，所以这里不显示假百分比。"}
        </Text>
        {hasRealPercent ? (
          <ProgressBar className="truth-progress" thickness="large" value={progressValue} />
        ) : (
          <div className="truth-progress-placeholder" aria-label="等待真实进度">
            <span />
          </div>
        )}
        <div className="open3d-progress-meta">
          <Metric label="阶段" value={status.phase || "--"} />
          <Metric label="文件" value={fileText} />
          <Metric label="已读点数" value={pointsText} />
          <Metric label="渲染点数" value={formatOptionalNumber(status.display_points)} />
        </div>
        {status.error ? <Text className="error-text">{status.error}</Text> : null}
        <div className="open3d-progress-actions">
          <Button onClick={props.onStop}>停止并切回标注画布</Button>
        </div>
        <Text size={200} className="mono-text">日志：{status.log_path || "--"}</Text>
      </div>
    </div>
  );
}

function formatVector(values: number[]) {
  return values.map((value) => value.toFixed(3)).join(", ");
}

function formatOptionalNumber(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "--";
  }
  return value.toLocaleString();
}

function lastPathPart(path: string) {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}

function boundsFromPreview(preview: PreviewData): Bounds2d {
  return {
    x_min: preview.bounds.minimum[0],
    x_max: preview.bounds.maximum[0],
    y_min: preview.bounds.minimum[1],
    y_max: preview.bounds.maximum[1]
  };
}

function boundsFromPointInfo(info: PointCloudInfo): Bounds2d | null {
  if (info.bounds.minimum.length < 2 || info.bounds.maximum.length < 2) {
    return null;
  }
  return {
    x_min: info.bounds.minimum[0],
    x_max: info.bounds.maximum[0],
    y_min: info.bounds.minimum[1],
    y_max: info.bounds.maximum[1]
  };
}

function boundsArea(bounds: Bounds2d) {
  return Math.max(bounds.x_max - bounds.x_min, 1e-6) * Math.max(bounds.y_max - bounds.y_min, 1e-6);
}

function boundsContains(outer: Bounds2d, inner: Bounds2d) {
  const marginX = Math.max(inner.x_max - inner.x_min, 1e-6) * 0.08;
  const marginY = Math.max(inner.y_max - inner.y_min, 1e-6) * 0.08;
  return (
    outer.x_min <= inner.x_min - marginX &&
    outer.x_max >= inner.x_max + marginX &&
    outer.y_min <= inner.y_min - marginY &&
    outer.y_max >= inner.y_max + marginY
  );
}

function expandBounds(bounds: Bounds2d, paddingRatio: number, limit: Bounds2d) {
  const spanX = Math.max(bounds.x_max - bounds.x_min, 1e-6);
  const spanY = Math.max(bounds.y_max - bounds.y_min, 1e-6);
  const padX = spanX * paddingRatio;
  const padY = spanY * paddingRatio;
  return {
    x_min: Math.max(limit.x_min, bounds.x_min - padX),
    x_max: Math.min(limit.x_max, bounds.x_max + padX),
    y_min: Math.max(limit.y_min, bounds.y_min - padY),
    y_max: Math.min(limit.y_max, bounds.y_max + padY)
  };
}

function boundsKey(bounds: Bounds2d) {
  return [bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max].map((value) => value.toFixed(2)).join(":");
}

function pointInsideBounds(point: [number, number], bounds: Bounds2d) {
  return point[0] >= bounds.x_min && point[0] <= bounds.x_max && point[1] >= bounds.y_min && point[1] <= bounds.y_max;
}

function guidedTargetToKey(target: GuidedTarget) {
  return `${target.kind}:${target.id}`;
}

function guidedTargetFromKey(key: string): GuidedTarget {
  const [kind, rawId] = key.split(":");
  const id = Math.max(1, Number(rawId) || 1);
  if (kind === "turnout_main") {
    return { kind: "turnout_main", id };
  }
  if (kind === "turnout_branch") {
    return { kind: "turnout_branch", id };
  }
  return { kind: "track", id };
}

function extractResultOverlay(result: ApiResult): ResultOverlay | null {
  const overlay = result.overlay as ResultOverlay | undefined;
  if (!overlay || !Array.isArray(overlay.tracks) || !Array.isArray(overlay.turnouts)) {
    return null;
  }
  return overlay;
}

function compactResultForDisplay(result: ApiResult): ApiResult {
  const overlay = extractResultOverlay(result);
  if (!overlay) {
    return result;
  }
  const displayResult: ApiResult = { ...result };
  delete displayResult.overlay;
  return {
    ...displayResult,
    overlay: {
      track_count: overlay.track_count,
      turnout_count: overlay.turnout_count,
      centerline_points: overlay.centerline_xy.length,
      rail_preview_points: overlay.rail_points_xy.length,
      tracks: overlay.tracks.map((track) => ({
        id: track.id,
        label: track.label,
        confidence: track.confidence,
        centerline_points: track.centerline_xy.length,
        rail_preview_points: track.rail_points_xy.length
      }))
    }
  };
}

function compactDomPipelineStatus(status: DomPipelineStatus): ApiResult {
  return {
    job_id: status.job_id,
    state: status.state,
    message: status.message,
    stage: status.stage_name,
    stage_status: status.stage_status,
    progress_percent: status.percent,
    out_dir: status.out_dir,
    outputs: status.outputs ?? {},
    summary_path: status.summary_path,
    log_path: status.log_path,
    progress_path: status.progress_path,
    error: status.error
  };
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
