import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { BackendClient, BackendConfig, DomPipelineStatus, HealthResponse, RasterProbe, SystemDevices } from "./lib/api";
import { lastPathPart, formatPercent, formatDuration, timeStamp, clamp } from "./lib/format";
import { Button } from "./components/Button";
import { Field, TextInput, inputClass } from "./components/Field";
import { Card } from "./components/Card";
import { Badge } from "./components/Badge";
import { Toggle } from "./components/Toggle";
import { ProgressBar } from "./components/ProgressBar";
import { Stepper, StepItem, StepState } from "./components/Stepper";
import { ThemeSwitch } from "./components/ThemeSwitch";
import { useTheme } from "./lib/theme";

type FormState = {
  domPath: string;
  modelPath: string;
  outputDir: string;
  dsmPath: string;
  lasDir: string;
  device: "cuda" | "cpu";
  threshold: string;
  maxTiles: string;
  force: boolean;
};

const initialForm: FormState = {
  domPath: "",
  modelPath: "",
  outputDir: "",
  dsmPath: "",
  lasDir: "",
  device: "cuda",
  threshold: "0.50",
  maxTiles: "0",
  force: false
};

type LogEntry = {
  id: number;
  time: string;
  level: "info" | "warn" | "error" | "success";
  text: string;
};

const STORAGE_KEY = "rce-form-v1";

export function App() {
  if (!window.railCurve) {
    return <BridgeError />;
  }

  const [backend, setBackend] = useState<BackendConfig | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [form, setForm] = useState<FormState>(() => loadFormFromStorage());
  const [pipelineJobId, setPipelineJobId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<DomPipelineStatus | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [pipelineStartTs, setPipelineStartTs] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const logIdRef = useRef(0);
  const { mode: themeMode, setMode: setThemeMode } = useTheme();

  // DOM CRS auto-detection.
  const [domProbe, setDomProbe] = useState<RasterProbe | null>(null);
  const [domProbeError, setDomProbeError] = useState<string | null>(null);
  const [domProbing, setDomProbing] = useState(false);
  const [epsgOverride, setEpsgOverride] = useState<string>(""); // only set when user manually overrides

  // Local hardware probe (CPU / GPU / PyTorch CUDA build).
  const [devices, setDevices] = useState<SystemDevices | null>(null);

  const client = useMemo(() => (backend ? new BackendClient(backend) : null), [backend]);

  // Persist form to localStorage so users don't retype paths every restart.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
    } catch {
      /* ignore */
    }
  }, [form]);

  // Acquire backend config once.
  useEffect(() => {
    void window.railCurve.backendConfig().then((config) => {
      setBackend(config);
      pushLog("info", `本地后端：${config.baseUrl}`);
    });
  }, []);

  // Tick "now" for live duration display.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  // Health probe every 2s, also serves as backend-up signal.
  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    const probe = async () => {
      try {
        const res = await client.health();
        if (cancelled) return;
        setHealth((prev) => {
          if (!prev?.ok) {
            pushLog("success", `后端已就绪 · Python ${res.python}`);
          }
          return res;
        });
      } catch {
        if (!cancelled) setHealth(null);
      }
    };
    void probe();
    const id = window.setInterval(probe, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client]);

  // Auto-detect CRS from DOM whenever domPath changes (debounced 300ms).
  useEffect(() => {
    if (!client) return;
    const path = form.domPath.trim();
    if (!path) {
      setDomProbe(null);
      setDomProbeError(null);
      setEpsgOverride("");
      return;
    }
    let cancelled = false;
    setDomProbing(true);
    setDomProbeError(null);
    const timer = window.setTimeout(async () => {
      try {
        const probe = await client.probeRaster(path);
        if (cancelled) return;
        setDomProbe(probe);
        setEpsgOverride("");
        if (probe.epsg) {
          pushLog("info", `DOM 自带坐标系：EPSG:${probe.epsg}（${probe.crs ?? ""}）`);
        } else {
          pushLog("warn", "DOM 文件没有写入坐标系，需要手动指定 EPSG");
        }
      } catch (err) {
        if (cancelled) return;
        setDomProbe(null);
        setDomProbeError((err as Error).message);
      } finally {
        if (!cancelled) setDomProbing(false);
      }
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, form.domPath]);

  // One-shot system devices probe once backend is up.
  useEffect(() => {
    if (!client || !health?.ok || devices) return;
    let cancelled = false;
    void (async () => {
      try {
        const result = await client.systemDevices();
        if (cancelled) return;
        setDevices(result);
        const gpuLabel =
          result.cuda.available && result.cuda.gpus.length > 0
            ? result.cuda.gpus.map((g) => g.name).join(", ")
            : "无 NVIDIA GPU";
        pushLog("info", `本机硬件：${result.cpu.name} · ${gpuLabel}`);
        if (!result.torch.installed) {
          pushLog("warn", "PyTorch 未安装：无法运行语义分割推理");
        } else if (!result.torch.cuda_runtime_available && result.cuda.available) {
          pushLog(
            "warn",
            `检测到 GPU 但 PyTorch CUDA 不可用（已装版本 ${result.torch.version}），需要装 CUDA 版 PyTorch`
          );
        }
      } catch (err) {
        if (!cancelled) {
          pushLog("warn", `读取硬件信息失败：${(err as Error).message}`);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, health?.ok]);

  // Poll pipeline status while running.
  useEffect(() => {
    if (!client || !pipelineJobId) return;
    let cancelled = false;
    let running = true;

    const poll = async () => {
      try {
        const status = await client.pipelineStatus(pipelineJobId);
        if (cancelled) return;
        setPipelineStatus((previous) => {
          if (previous?.state !== status.state) {
            if (status.state === "completed") {
              pushLog("success", `流水线完成：${status.out_dir}`);
            } else if (status.state === "failed") {
              pushLog("error", status.error || status.message || "流水线失败");
            } else if (status.state === "stopped") {
              pushLog("warn", "流水线已停止");
            }
          }
          if (
            previous?.stage_name !== status.stage_name &&
            status.stage_name &&
            status.state === "running"
          ) {
            pushLog(
              "info",
              `阶段：${status.stage_name}${status.stage_description ? ` · ${status.stage_description}` : ""}`
            );
          }
          // Stop polling once backend is no longer running.
          if (status.running === false) {
            running = false;
          }
          return status;
        });
      } catch (err) {
        if (!cancelled) {
          pushLog("warn", `状态轮询失败：${(err as Error).message}`);
        }
      }
    };

    void poll();
    const id = window.setInterval(() => {
      if (running) void poll();
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, pipelineJobId]);

  function pushLog(level: LogEntry["level"], text: string) {
    setLogs((existing) => {
      logIdRef.current += 1;
      return [
        { id: logIdRef.current, time: timeStamp(), level, text },
        ...existing
      ].slice(0, 400);
    });
  }

  const updateForm = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  }, []);

  const formIsValid = useMemo(() => {
    return (
      form.domPath.trim().length > 0 &&
      form.modelPath.trim().length > 0 &&
      form.outputDir.trim().length > 0
    );
  }, [form]);

  // Resolve which EPSG to actually send: file's own CRS first, then user override.
  const effectiveEpsg: number | null = useMemo(() => {
    const overrideNum = parseInt(epsgOverride, 10);
    if (Number.isFinite(overrideNum) && overrideNum > 0) return overrideNum;
    if (domProbe?.epsg) return domProbe.epsg;
    return null;
  }, [domProbe, epsgOverride]);

  // EPSG-related run blockers: if DOM has no CRS and user hasn't overridden, can't start.
  const epsgReady =
    !form.domPath.trim() ||
    domProbing ||
    !!domProbe?.epsg ||
    (Number.isFinite(parseInt(epsgOverride, 10)) && parseInt(epsgOverride, 10) > 0);

  const pipelineRunning =
    !!pipelineStatus &&
    (pipelineStatus.running ||
      ["starting", "planned", "running"].includes(pipelineStatus.state));
  const canStart = formIsValid && epsgReady && !!health?.ok && !pipelineRunning && !busy;

  async function handleStart() {
    if (!client) return;
    if (!effectiveEpsg) {
      pushLog("error", "无法确定坐标系：DOM 没有 CRS，请填写 EPSG 覆盖。");
      return;
    }
    setBusy(true);
    try {
      pushLog("info", `提交 DOM 流水线 · EPSG:${effectiveEpsg}`);
      const status = await client.startPipeline({
        domPath: form.domPath.trim(),
        modelPath: form.modelPath.trim(),
        outputDir: form.outputDir.trim(),
        dsmPath: form.dsmPath,
        lasDir: form.lasDir,
        device: form.device,
        threshold: parseFloat(form.threshold) || 0.5,
        maxTiles: parseInt(form.maxTiles, 10) || 0,
        force: form.force,
        epsg: effectiveEpsg
      });
      setPipelineJobId(status.job_id);
      setPipelineStatus(status);
      setPipelineStartTs(Date.now());
      pushLog("success", `已启动 · job ${status.job_id}`);
    } catch (err) {
      pushLog("error", (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleStop() {
    if (!client || !pipelineJobId) return;
    setBusy(true);
    try {
      const status = await client.stopPipeline(pipelineJobId);
      setPipelineStatus(status);
      pushLog("warn", "已请求停止流水线");
    } catch (err) {
      pushLog("error", (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // File pickers
  async function pickDom() {
    const path = await window.railCurve.openDomDialog();
    if (path) updateForm("domPath", path);
  }
  async function pickModel() {
    const path = await window.railCurve.openModelDialog();
    if (path) updateForm("modelPath", path);
  }
  async function pickDsm() {
    const path = await window.railCurve.openDsmDialog();
    if (path) updateForm("dsmPath", path);
  }
  async function pickLasDir() {
    const path = await window.railCurve.openLasDirectoryDialog();
    if (path) updateForm("lasDir", path);
  }
  async function pickOutput() {
    const path = await window.railCurve.selectOutputDirectory();
    if (path) updateForm("outputDir", path);
  }

  return (
    <div className="flex flex-col h-full bg-[var(--color-canvas)]">
      <TopBar
        backend={backend}
        health={health}
        pipelineRunning={pipelineRunning}
        themeMode={themeMode}
        onThemeChange={setThemeMode}
      />

      <div className="flex flex-1 min-h-0">
        <SidePanel
          form={form}
          updateForm={updateForm}
          canStart={canStart}
          busy={busy}
          pipelineRunning={pipelineRunning}
          onStart={handleStart}
          onStop={handleStop}
          domProbe={domProbe}
          domProbing={domProbing}
          domProbeError={domProbeError}
          epsgOverride={epsgOverride}
          setEpsgOverride={setEpsgOverride}
          devices={devices}
          pickers={{
            dom: pickDom,
            model: pickModel,
            dsm: pickDsm,
            lasDir: pickLasDir,
            output: pickOutput
          }}
        />

        <main className="flex-1 min-w-0 flex flex-col gap-4 p-4 overflow-auto">
          <PipelineMonitor
            status={pipelineStatus}
            startedAt={pipelineStartTs}
            now={now}
          />

          <div className="grid grid-cols-1 xl:grid-cols-[3fr_2fr] gap-4 flex-1 min-h-0">
            <LogPanel logs={logs} onClear={() => setLogs([])} />
            <OutputsPanel status={pipelineStatus} />
          </div>
        </main>
      </div>
    </div>
  );
}

/* ---------------------------------- Top bar --------------------------------- */

function TopBar({
  backend,
  health,
  pipelineRunning,
  themeMode,
  onThemeChange
}: {
  backend: BackendConfig | null;
  health: HealthResponse | null;
  pipelineRunning: boolean;
  themeMode: "system" | "light" | "dark";
  onThemeChange: (next: "system" | "light" | "dark") => void;
}) {
  return (
    <header className="flex items-center justify-between gap-4 h-14 px-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
      <div className="flex items-center gap-3 min-w-0">
        <Logo />
        <div className="flex flex-col min-w-0">
          <span className="text-sm font-semibold tracking-tight">Rail Curve Extractor</span>
          <span className="text-[11px] text-[var(--color-text-dim)] truncate">
            DOM → 语义分割 → 后处理 → 3D 中心线
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {pipelineRunning ? (
          <Badge tone="info" dot>
            流水线运行中
          </Badge>
        ) : null}
        <Badge tone={health?.ok ? "success" : "warn"} dot>
          {health?.ok ? `后端在线 · Python ${health.python}` : "等待后端"}
        </Badge>
        <span className="text-[11px] font-mono text-[var(--color-text-dim)]">
          {backend?.baseUrl ?? "—"}
        </span>
        <div className="w-px h-5 bg-[var(--color-border)] mx-1" />
        <ThemeSwitch mode={themeMode} onChange={onThemeChange} />
      </div>
    </header>
  );
}

function Logo() {
  return (
    <div className="flex items-center justify-center w-8 h-8 rounded-md bg-[var(--color-accent-soft)] text-[var(--color-accent)]">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M5 21 12 3l7 18" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M8.5 13h7" strokeLinecap="round" />
      </svg>
    </div>
  );
}

/* --------------------------------- Side panel ------------------------------- */

type SidePanelProps = {
  form: FormState;
  updateForm: <K extends keyof FormState>(key: K, value: FormState[K]) => void;
  canStart: boolean;
  busy: boolean;
  pipelineRunning: boolean;
  onStart: () => void;
  onStop: () => void;
  domProbe: RasterProbe | null;
  domProbing: boolean;
  domProbeError: string | null;
  epsgOverride: string;
  setEpsgOverride: (value: string) => void;
  devices: SystemDevices | null;
  pickers: {
    dom: () => Promise<void>;
    model: () => Promise<void>;
    dsm: () => Promise<void>;
    lasDir: () => Promise<void>;
    output: () => Promise<void>;
  };
};

function SidePanel({
  form,
  updateForm,
  canStart,
  busy,
  pipelineRunning,
  onStart,
  onStop,
  domProbe,
  domProbing,
  domProbeError,
  epsgOverride,
  setEpsgOverride,
  devices,
  pickers
}: SidePanelProps) {
  const inputsDisabled = pipelineRunning;
  return (
    <aside className="w-[380px] shrink-0 border-r border-[var(--color-border)] bg-[var(--color-surface)] flex flex-col">
      <div className="flex-1 min-h-0 overflow-auto p-4 flex flex-col gap-4">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">输入</h2>
          <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5">
            必填三项：DOM、权重、输出目录。其它可缺省。
          </p>
        </div>

        <PathPicker
          label="DOM 影像"
          required
          value={form.domPath}
          placeholder="dom.tif / dom.jpg"
          onChange={(value) => updateForm("domPath", value)}
          onPick={pickers.dom}
          disabled={inputsDisabled}
        />

        <CrsStatus
          domPath={form.domPath}
          probe={domProbe}
          probing={domProbing}
          error={domProbeError}
          epsgOverride={epsgOverride}
          setEpsgOverride={setEpsgOverride}
          disabled={inputsDisabled}
        />

        <PathPicker
          label="DeepLab 权重"
          required
          value={form.modelPath}
          placeholder="rail_semantic_deeplab_resnet50.pt"
          onChange={(value) => updateForm("modelPath", value)}
          onPick={pickers.model}
          disabled={inputsDisabled}
        />

        <PathPicker
          label="输出目录"
          required
          value={form.outputDir}
          placeholder="例：D:\output\dom_centerline_v1"
          onChange={(value) => updateForm("outputDir", value)}
          onPick={pickers.output}
          disabled={inputsDisabled}
        />

        <div className="h-px bg-[var(--color-border)] my-1" />

        <div>
          <h3 className="text-xs font-semibold tracking-tight text-[var(--color-text-muted)]">3D 高程（可选）</h3>
          <p className="text-[11px] text-[var(--color-text-dim)] mt-0.5">缺省时只产 2D 中心线。</p>
        </div>

        <PathPicker
          label="DSM 栅格"
          value={form.dsmPath}
          placeholder="dsm.tif（可空）"
          onChange={(value) => updateForm("dsmPath", value)}
          onPick={pickers.dsm}
          disabled={inputsDisabled}
        />

        <PathPicker
          label="LAS / LAZ 目录"
          value={form.lasDir}
          placeholder="点云目录（可空）"
          onChange={(value) => updateForm("lasDir", value)}
          onPick={pickers.lasDir}
          disabled={inputsDisabled}
        />

        <div className="h-px bg-[var(--color-border)] my-1" />

        <div>
          <h3 className="text-xs font-semibold tracking-tight text-[var(--color-text-muted)]">运行参数</h3>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Field label="推理设备" className="col-span-2">
            <DevicePicker
              devices={devices}
              value={form.device}
              onChange={(value) => updateForm("device", value)}
              disabled={inputsDisabled}
            />
          </Field>

          <Field label="分割阈值" hint="0–1">
            <TextInput
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={form.threshold}
              onChange={(event) => updateForm("threshold", event.currentTarget.value)}
              disabled={inputsDisabled}
              mono
            />
          </Field>

          <Field label="最大瓦片" hint="0 = 不限制">
            <TextInput
              type="number"
              min={0}
              value={form.maxTiles}
              onChange={(event) => updateForm("maxTiles", event.currentTarget.value)}
              disabled={inputsDisabled}
              mono
            />
          </Field>
        </div>

        <Toggle
          checked={form.force}
          onChange={(value) => updateForm("force", value)}
          label="强制重跑"
          hint="忽略已有阶段产物，全部重新计算"
          disabled={inputsDisabled}
        />
      </div>

      <div className="border-t border-[var(--color-border)] p-3 flex gap-2">
        {pipelineRunning ? (
          <Button variant="danger" className="flex-1" onClick={onStop} loading={busy}>
            停止流水线
          </Button>
        ) : (
          <Button
            variant="primary"
            className="flex-1"
            disabled={!canStart}
            loading={busy}
            onClick={onStart}
          >
            {busy ? "提交中" : "开始流水线"}
          </Button>
        )}
      </div>
    </aside>
  );
}

function PathPicker({
  label,
  required,
  value,
  placeholder,
  onChange,
  onPick,
  disabled
}: {
  label: string;
  required?: boolean;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  onPick: () => void | Promise<void>;
  disabled?: boolean;
}) {
  return (
    <Field label={label} required={required}>
      <div className="flex gap-2">
        <TextInput
          value={value}
          onChange={(event) => onChange(event.currentTarget.value)}
          placeholder={placeholder}
          disabled={disabled}
          mono
          className="flex-1"
        />
        <Button onClick={() => void onPick()} disabled={disabled} variant="secondary" size="md">
          选择
        </Button>
      </div>
    </Field>
  );
}

function DevicePicker({
  devices,
  value,
  onChange,
  disabled
}: {
  devices: SystemDevices | null;
  value: FormState["device"];
  onChange: (value: FormState["device"]) => void;
  disabled?: boolean;
}) {
  // Resolve concrete labels.
  const cudaUsable = !!devices?.cuda.available && !!devices?.torch.cuda_runtime_available;
  const gpuName =
    devices?.cuda.gpus[0]?.name ?? (devices?.cuda.available ? "GPU 已检测到" : "未检测到 NVIDIA GPU");
  const gpuMemoryGb = devices?.cuda.gpus[0]?.memory_total_mib
    ? `${(devices.cuda.gpus[0].memory_total_mib / 1024).toFixed(0)} GB`
    : null;
  const cpuName = devices?.cpu.name ?? "CPU";
  const cpuCores =
    devices?.cpu.physical_cores && devices?.cpu.logical_cores
      ? `${devices.cpu.physical_cores} 核 / ${devices.cpu.logical_cores} 线程`
      : devices?.cpu.logical_cores
        ? `${devices.cpu.logical_cores} 线程`
        : null;

  // Auto-correct invalid selection: if user has cuda saved but cuda is unusable, switch to cpu.
  useEffect(() => {
    if (devices && value === "cuda" && !cudaUsable) {
      onChange("cpu");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [devices, cudaUsable]);

  return (
    <div className="flex flex-col gap-2 w-full">
      <div className="grid grid-cols-2 gap-2">
        <DeviceCard
          active={value === "cuda"}
          disabled={disabled || !cudaUsable}
          icon={<GpuIcon />}
          title="CUDA"
          subtitle={gpuName}
          meta={gpuMemoryGb ?? (devices ? "—" : "检测中…")}
          onClick={() => cudaUsable && onChange("cuda")}
        />
        <DeviceCard
          active={value === "cpu"}
          disabled={disabled}
          icon={<CpuIcon />}
          title="CPU"
          subtitle={cpuName}
          meta={cpuCores ?? (devices ? "—" : "检测中…")}
          onClick={() => onChange("cpu")}
        />
      </div>

      <DeviceAdvice devices={devices} />
    </div>
  );
}

function DeviceCard({
  active,
  disabled,
  icon,
  title,
  subtitle,
  meta,
  onClick
}: {
  active: boolean;
  disabled?: boolean;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  meta: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "flex flex-col gap-1.5 p-3 rounded-md border text-left transition-colors min-w-0",
        active
          ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
          : "border-[var(--color-border)] bg-[var(--color-surface-2)] hover:border-[var(--color-border-strong)]",
        disabled && "opacity-45 cursor-not-allowed hover:border-[var(--color-border)]"
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={clsx(
            "shrink-0",
            active ? "text-[var(--color-accent)]" : "text-[var(--color-text-muted)]"
          )}
        >
          {icon}
        </span>
        <span
          className={clsx(
            "text-xs font-semibold tracking-tight",
            active ? "text-[var(--color-accent)]" : "text-[var(--color-text)]"
          )}
        >
          {title}
        </span>
      </div>
      <div className="text-[11px] text-[var(--color-text-muted)] truncate" title={subtitle}>
        {subtitle}
      </div>
      <div className="text-[10px] text-[var(--color-text-dim)] font-mono">{meta}</div>
    </button>
  );
}

function DeviceAdvice({ devices }: { devices: SystemDevices | null }) {
  if (!devices) return null;

  const { cuda, torch } = devices;

  if (!torch.installed) {
    return (
      <Hint tone="danger">
        PyTorch 未安装，无法运行语义分割推理。
        <span className="block mt-0.5 text-[var(--color-text-dim)]">
          安装方式：根据 GPU 情况执行 <code className="font-mono text-[10px]">pip install torch torchvision</code>
          （CUDA 版可在 pytorch.org 选择对应轮子）。
        </span>
      </Hint>
    );
  }

  if (cuda.available && !torch.cuda_runtime_available) {
    return (
      <Hint tone="warn">
        检测到 NVIDIA GPU，但当前 PyTorch（{torch.version}）是 CPU 版本。
        <span className="block mt-0.5 text-[var(--color-text-dim)]">
          安装 GPU 版后才能用 CUDA 推理：从 pytorch.org 选 CUDA {cuda.gpus[0]?.driver_version ? "对应版本" : "12.x"} 的轮子。
        </span>
      </Hint>
    );
  }

  if (!cuda.available) {
    // No NVIDIA driver / device. Make it explicit AMD/Intel users land on CPU.
    return (
      <Hint tone="info">
        未检测到 NVIDIA 显卡（或驱动未装），将使用 CPU 推理。
        <span className="block mt-0.5 text-[var(--color-text-dim)]">
          目前 PyTorch CUDA 仅支持 NVIDIA。AMD/Intel 显卡在 Windows 上无法直接加速；如需 GPU 加速，建议换 NVIDIA 卡或在 Linux 上配置 ROCm。
        </span>
      </Hint>
    );
  }

  // CUDA usable: low-key confirmation only.
  return (
    <Hint tone="success">
      已检测到 GPU 推理环境：CUDA {torch.cuda_build} · PyTorch {torch.version}
    </Hint>
  );
}

function Hint({
  tone,
  children
}: {
  tone: "success" | "info" | "warn" | "danger";
  children: React.ReactNode;
}) {
  const toneClass = {
    success: "border-[var(--color-accent)]/30 bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
    info: "border-[var(--color-info)]/30 bg-[var(--color-info)]/10 text-[var(--color-info)]",
    warn: "border-[var(--color-warn)]/30 bg-[var(--color-warn)]/10 text-[var(--color-warn)]",
    danger: "border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 text-[var(--color-danger)]"
  }[tone];
  return (
    <div className={clsx("rounded-md border px-2.5 py-2 text-[11px] leading-relaxed", toneClass)}>
      {children}
    </div>
  );
}

function GpuIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="2" y="6" width="20" height="12" rx="2" />
      <circle cx="9" cy="12" r="2.5" />
      <circle cx="16" cy="12" r="1.5" />
      <path d="M2 10h2M2 14h2" />
    </svg>
  );
}

function CpuIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <rect x="9" y="9" width="6" height="6" rx="0.5" />
      <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
    </svg>
  );
}

function CrsStatus({
  domPath,
  probe,
  probing,
  error,
  epsgOverride,
  setEpsgOverride,
  disabled
}: {
  domPath: string;
  probe: RasterProbe | null;
  probing: boolean;
  error: string | null;
  epsgOverride: string;
  setEpsgOverride: (value: string) => void;
  disabled?: boolean;
}) {
  if (!domPath.trim()) return null;

  let body: React.ReactNode;
  let tone: "info" | "success" | "warn" | "danger" = "info";
  let label = "坐标系";

  if (probing) {
    tone = "info";
    label = "读取中";
    body = <span className="text-[var(--color-text-muted)]">正在读取 DOM 自带坐标系…</span>;
  } else if (error) {
    tone = "danger";
    label = "读取失败";
    body = (
      <span className="text-[var(--color-danger)] font-mono text-[11px] break-all">{error}</span>
    );
  } else if (probe?.epsg) {
    tone = "success";
    label = `EPSG:${probe.epsg}`;
    body = (
      <span className="text-[var(--color-text-muted)] truncate" title={probe.crs ?? ""}>
        {probe.crs ?? "—"}
      </span>
    );
  } else if (probe) {
    tone = "warn";
    label = "无 CRS";
    body = (
      <div className="flex flex-col gap-1.5">
        <span className="text-[var(--color-text-muted)]">
          这个 DOM 没有写入坐标系信息，需要手动指定。
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-[var(--color-text-dim)]">EPSG</span>
          <TextInput
            mono
            value={epsgOverride}
            onChange={(event) => setEpsgOverride(event.currentTarget.value)}
            placeholder="例如 32651"
            disabled={disabled}
            className="h-7 w-28 text-xs"
          />
        </div>
      </div>
    );
  } else {
    return null;
  }

  return (
    <div className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <Badge tone={tone} dot>
          {label}
        </Badge>
        {probe?.width && probe?.height ? (
          <span className="text-[10px] font-mono text-[var(--color-text-dim)]">
            {probe.width}×{probe.height} · {probe.driver}
          </span>
        ) : null}
      </div>
      <div className="text-[11px] leading-relaxed">{body}</div>
    </div>
  );
}

/* ---------------------------- Pipeline monitor card ------------------------- */

const STAGE_BLUEPRINT: Array<{ id: string; label: string; matchers: RegExp[] }> = [
  { id: "tile", label: "切瓦片", matchers: [/tile/i, /切瓦片/, /切片/] },
  { id: "infer", label: "语义分割", matchers: [/infer/i, /predict/i, /seg/i, /语义|分割|推理/] },
  { id: "post", label: "后处理", matchers: [/post/i, /topology/i, /skeleton/i, /后处理|拓扑|骨架/] },
  { id: "package", label: "中心线打包", matchers: [/package/i, /strict/i, /中心线|打包/] },
  { id: "z", label: "Z 补全", matchers: [/z/i, /add_z/i, /lift/i, /高程|补Z/] },
  { id: "deliver", label: "交付", matchers: [/deliver/i, /shapefile/i, /export/i, /交付|导出/] }
];

function inferStepStates(status: DomPipelineStatus | null): StepItem[] {
  const steps: StepItem[] = STAGE_BLUEPRINT.map((blueprint) => ({
    id: blueprint.id,
    label: blueprint.label,
    state: "pending" as StepState
  }));

  if (!status) return steps;

  // Use stage_index/stage_count if backend provides reliable numeric progress.
  const idx = typeof status.stage_index === "number" ? status.stage_index : null;
  const total = typeof status.stage_count === "number" ? status.stage_count : null;

  // Find which blueprint stage matches current stage_name.
  const currentName = status.stage_name ?? "";
  let activeIdx = -1;
  if (currentName) {
    activeIdx = STAGE_BLUEPRINT.findIndex((s) => s.matchers.some((re) => re.test(currentName)));
  }

  // Fallback to numeric mapping when name doesn't match any blueprint.
  if (activeIdx < 0 && idx != null && total && total > 0) {
    activeIdx = Math.min(STAGE_BLUEPRINT.length - 1, Math.floor(((idx - 1) / total) * STAGE_BLUEPRINT.length));
  }

  if (status.state === "completed") {
    return steps.map((step) => ({ ...step, state: "done" }));
  }

  if (status.state === "failed") {
    const failAt = Math.max(0, activeIdx);
    return steps.map((step, i) => ({
      ...step,
      state: i < failAt ? "done" : i === failAt ? "failed" : "pending"
    }));
  }

  if (status.state === "stopped") {
    const stopAt = Math.max(0, activeIdx);
    return steps.map((step, i) => ({
      ...step,
      state: i < stopAt ? "done" : i === stopAt ? "skipped" : "pending"
    }));
  }

  if (activeIdx >= 0) {
    return steps.map((step, i) => ({
      ...step,
      state: i < activeIdx ? "done" : i === activeIdx ? "running" : "pending",
      detail: i === activeIdx ? currentName || undefined : undefined
    }));
  }

  // Running but no stage info yet
  if (status.state === "running" || status.state === "starting" || status.state === "planned") {
    return steps.map((step, i) => ({ ...step, state: i === 0 ? "running" : "pending" }));
  }

  return steps;
}

function PipelineMonitor({
  status,
  startedAt,
  now
}: {
  status: DomPipelineStatus | null;
  startedAt: number | null;
  now: number;
}) {
  const steps = useMemo(() => inferStepStates(status), [status]);
  const tone =
    status?.state === "failed"
      ? "danger"
      : status?.state === "completed"
        ? "success"
        : status?.state
          ? "info"
          : "neutral";
  const indeterminate =
    !!status &&
    status.running &&
    !(typeof status.percent === "number" && Number.isFinite(status.percent));

  const elapsed = startedAt ? now - startedAt : 0;

  return (
    <Card
      title="流水线监控"
      subtitle={
        status
          ? status.message || status.stage_description || "等待状态更新…"
          : "尚未启动 · 在左侧填写输入后点击开始"
      }
      actions={
        <Badge tone={tone} dot>
          {status?.state ?? "idle"}
        </Badge>
      }
      bodyClassName="flex flex-col gap-4"
    >
      <Stepper steps={steps} />

      <ProgressBar
        value={typeof status?.percent === "number" ? status.percent : null}
        indeterminate={indeterminate}
        tone={status?.state === "failed" ? "danger" : "accent"}
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="进度" value={formatPercent(status?.percent)} />
        <Stat
          label="阶段"
          value={
            status?.stage_index && status?.stage_count
              ? `${status.stage_index}/${status.stage_count}`
              : status?.stage_name ?? "—"
          }
        />
        <Stat label="耗时" value={status ? formatDuration(elapsed) : "—"} />
        <Stat label="进程" value={status?.pid ? `PID ${status.pid}` : "—"} mono />
      </div>

      {status?.error ? (
        <div className="rounded-md border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 px-3 py-2 text-xs text-[var(--color-danger)] font-mono whitespace-pre-wrap break-all">
          {status.error}
        </div>
      ) : null}
    </Card>
  );
}

function Stat({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">{label}</div>
      <div
        className={clsx(
          "mt-1 text-sm font-semibold truncate",
          mono ? "font-mono" : "",
          "text-[var(--color-text)]"
        )}
      >
        {value}
      </div>
    </div>
  );
}

/* ---------------------------------- Logs ------------------------------------ */

function LogPanel({
  logs,
  onClear
}: {
  logs: LogEntry[];
  onClear: () => void;
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!query.trim()) return logs;
    const lower = query.trim().toLowerCase();
    return logs.filter((entry) => entry.text.toLowerCase().includes(lower));
  }, [logs, query]);

  async function handleCopy() {
    const text = filtered
      .slice()
      .reverse()
      .map((entry) => `[${entry.time}] ${entry.text}`)
      .join("\n");
    await window.railCurve.writeClipboard(text);
  }

  return (
    <Card
      title="运行日志"
      subtitle="最新事件位于顶部 · 仅前端事件流，详细日志见输出目录"
      actions={
        <>
          <input
            value={query}
            onChange={(event) => setQuery(event.currentTarget.value)}
            placeholder="过滤"
            className={clsx(inputClass, "h-7 w-32 text-xs px-2")}
          />
          <Button size="sm" variant="ghost" onClick={handleCopy} disabled={filtered.length === 0}>
            复制
          </Button>
          <Button size="sm" variant="ghost" onClick={onClear} disabled={logs.length === 0}>
            清空
          </Button>
        </>
      }
      bodyClassName="p-0"
      className="min-h-0"
    >
      <div className="h-[320px] overflow-auto font-mono text-xs leading-relaxed">
        {filtered.length === 0 ? (
          <div className="h-full flex items-center justify-center text-[var(--color-text-dim)] text-xs">
            {logs.length === 0 ? "暂无日志" : "无匹配条目"}
          </div>
        ) : (
          <ul className="divide-y divide-[var(--color-border)]/60">
            {filtered.map((entry) => (
              <li key={entry.id} className="flex gap-3 px-4 py-1.5 hover:bg-[var(--color-surface-2)]">
                <span className="text-[var(--color-text-dim)] shrink-0">{entry.time}</span>
                <span className={clsx("shrink-0 w-12 uppercase text-[10px]", logLevelColor(entry.level))}>
                  {entry.level}
                </span>
                <span className="text-[var(--color-text)] break-all">{entry.text}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

function logLevelColor(level: LogEntry["level"]): string {
  switch (level) {
    case "success":
      return "text-[var(--color-accent)]";
    case "warn":
      return "text-[var(--color-warn)]";
    case "error":
      return "text-[var(--color-danger)]";
    default:
      return "text-[var(--color-info)]";
  }
}

/* ---------------------------------- Outputs --------------------------------- */

function OutputsPanel({ status }: { status: DomPipelineStatus | null }) {
  const outputs = status?.outputs ?? {};
  const entries = Object.entries(outputs)
    .filter(([, value]) => typeof value === "string" && value.length > 0)
    .map(([key, value]) => ({ key, value: String(value) }));

  const summaryPath = status?.summary_path;
  const outDir = status?.out_dir;
  const logPath = status?.log_path;

  async function reveal(path: string) {
    if (!path) return;
    await window.railCurve.revealInExplorer(path);
  }
  async function openFolder(path: string) {
    if (!path) return;
    await window.railCurve.openPath(path);
  }
  async function copy(text: string) {
    await window.railCurve.writeClipboard(text);
  }

  return (
    <Card
      title="产出"
      subtitle="完成后点击文件名可在资源管理器中定位"
      actions={
        outDir ? (
          <>
            <Button size="sm" variant="secondary" onClick={() => void openFolder(outDir)}>
              打开输出目录
            </Button>
          </>
        ) : null
      }
      bodyClassName="p-0"
      className="min-h-0"
    >
      <div className="h-[320px] overflow-auto">
        {!status ? (
          <div className="h-full flex items-center justify-center text-[var(--color-text-dim)] text-xs px-6 text-center">
            流水线完成后会在这里列出 2D / 3D Shapefile、证据图层和 manifest
          </div>
        ) : (
          <ul className="divide-y divide-[var(--color-border)]/60">
            {outDir ? (
              <OutputRow
                kind="目录"
                label="out_dir"
                path={outDir}
                onReveal={() => void openFolder(outDir)}
                onCopy={() => void copy(outDir)}
              />
            ) : null}
            {entries.length === 0 ? (
              <li className="px-4 py-3 text-xs text-[var(--color-text-dim)]">
                {status.state === "running" || status.state === "starting"
                  ? "运行中，等待阶段输出…"
                  : "暂无具名产出文件"}
              </li>
            ) : (
              entries.map((entry) => (
                <OutputRow
                  key={entry.key}
                  kind="文件"
                  label={entry.key}
                  path={entry.value}
                  onReveal={() => void reveal(entry.value)}
                  onCopy={() => void copy(entry.value)}
                />
              ))
            )}
            {summaryPath ? (
              <OutputRow
                kind="摘要"
                label="summary"
                path={summaryPath}
                onReveal={() => void reveal(summaryPath)}
                onCopy={() => void copy(summaryPath)}
              />
            ) : null}
            {logPath ? (
              <OutputRow
                kind="日志"
                label="pipeline.log"
                path={logPath}
                onReveal={() => void reveal(logPath)}
                onCopy={() => void copy(logPath)}
              />
            ) : null}
          </ul>
        )}
      </div>
    </Card>
  );
}

function OutputRow({
  kind,
  label,
  path,
  onReveal,
  onCopy
}: {
  kind: string;
  label: string;
  path: string;
  onReveal: () => void;
  onCopy: () => void;
}) {
  return (
    <li className="group flex items-center gap-3 px-4 py-2 hover:bg-[var(--color-surface-2)]">
      <Badge tone="neutral" className="shrink-0">
        {kind}
      </Badge>
      <div className="min-w-0 flex-1">
        <div className="text-xs font-medium text-[var(--color-text)] truncate">{label}</div>
        <div className="text-[11px] font-mono text-[var(--color-text-dim)] truncate" title={path}>
          {path}
        </div>
      </div>
      <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1 shrink-0">
        <Button size="sm" variant="ghost" onClick={onCopy}>
          复制
        </Button>
        <Button size="sm" variant="ghost" onClick={onReveal}>
          定位
        </Button>
      </div>
    </li>
  );
}

/* ----------------------------- Bridge error fallback ------------------------ */

function BridgeError() {
  return (
    <div className="flex items-center justify-center h-full p-8">
      <div className="max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
        <h1 className="text-lg font-semibold text-[var(--color-danger)]">Electron 桥接未加载</h1>
        <p className="mt-2 text-sm text-[var(--color-text-muted)] leading-relaxed">
          preload 脚本没有暴露 <code className="font-mono">window.railCurve</code>。请重新执行
          <code className="font-mono"> npm run build </code>
          后再启动应用。
        </p>
      </div>
    </div>
  );
}

/* ----------------------------- Local persistence ---------------------------- */

function loadFormFromStorage(): FormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return initialForm;
    const parsed = JSON.parse(raw) as Partial<FormState>;
    return {
      ...initialForm,
      ...parsed,
      device: parsed.device === "cpu" ? "cpu" : "cuda",
      threshold: typeof parsed.threshold === "string" ? parsed.threshold : initialForm.threshold,
      maxTiles: typeof parsed.maxTiles === "string" ? parsed.maxTiles : initialForm.maxTiles,
      force: !!parsed.force
    };
  } catch {
    return initialForm;
  }
}

// suppress unused-import lint when only types are referenced
void clamp;
void lastPathPart;
