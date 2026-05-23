// Thin client around the local FastAPI backend exposed via ipc bridge.
// Backend contract reference: src/rail_curve_extractor/backend/app.py

export type BackendConfig = {
  baseUrl: string;
  token: string;
};

export type HealthResponse = {
  ok: boolean;
  service: string;
  python: string;
};

export type DomPipelineStatus = {
  job_id: string;
  state: "starting" | "planned" | "running" | "completed" | "failed" | "stopped" | string;
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

export type StartPipelineParams = {
  domPath: string;
  modelPath: string;
  outputDir: string;
  dsmPath?: string;
  lasDir?: string;
  device: "cuda" | "cpu" | string;
  threshold: number;
  maxTiles: number;
  force: boolean;
  epsg: number;
};

export type RasterProbe = {
  path: string;
  epsg: number | null;
  crs: string | null;
  width: number;
  height: number;
  band_count: number;
  driver: string;
  bounds: { left: number; bottom: number; right: number; top: number };
  pixel_size: [number, number];
};

export type CpuInfo = {
  name: string;
  arch: string;
  logical_cores: number | null;
  physical_cores: number | null;
  platform: string;
};

export type GpuInfo = {
  name: string;
  memory_total_mib: number | null;
  driver_version: string | null;
  compute_capability: string | null;
};

export type CudaInfo = {
  available: boolean;
  reason: "missing-driver" | "smi-failed" | "no-device" | null;
  message: string | null;
  gpus: GpuInfo[];
};

export type TorchInfo = {
  installed: boolean;
  version: string | null;
  cuda_build: string | null;
  cuda_runtime_available: boolean;
  device_count: number;
  message: string | null;
};

export type SystemDevices = {
  cpu: CpuInfo;
  cuda: CudaInfo;
  torch: TorchInfo;
};

export class BackendClient {
  constructor(private readonly config: BackendConfig) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.config.baseUrl}${path}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        "x-local-token": this.config.token,
        ...(init.headers ?? {})
      }
    });
    const text = await response.text();
    let payload: unknown = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text };
      }
    }
    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && payload !== null && "detail" in payload
          ? String((payload as { detail?: unknown }).detail ?? "")
          : "";
      throw new Error(detail || `请求失败 ${response.status}`);
    }
    return payload as T;
  }

  health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/api/health");
  }

  probeRaster(path: string): Promise<RasterProbe> {
    return this.request<RasterProbe>("/api/raster/probe", {
      method: "POST",
      body: JSON.stringify({ path })
    });
  }

  systemDevices(): Promise<SystemDevices> {
    return this.request<SystemDevices>("/api/system/devices");
  }

  startPipeline(params: StartPipelineParams): Promise<DomPipelineStatus> {
    return this.request<DomPipelineStatus>("/api/dom-pipeline/start", {
      method: "POST",
      body: JSON.stringify({
        dom_path: params.domPath,
        model_path: params.modelPath,
        output_dir: params.outputDir,
        dsm_path: params.dsmPath?.trim() || null,
        las_dir: params.lasDir?.trim() || null,
        profile: "strict-auto",
        device: params.device || "cuda",
        threshold: params.threshold,
        max_tiles: Math.max(0, params.maxTiles | 0),
        force: params.force,
        epsg: params.epsg
      })
    });
  }

  pipelineStatus(jobId: string): Promise<DomPipelineStatus> {
    return this.request<DomPipelineStatus>(`/api/dom-pipeline/status/${jobId}`);
  }

  stopPipeline(jobId: string): Promise<DomPipelineStatus> {
    return this.request<DomPipelineStatus>(`/api/dom-pipeline/stop/${jobId}`, {
      method: "POST"
    });
  }
}
