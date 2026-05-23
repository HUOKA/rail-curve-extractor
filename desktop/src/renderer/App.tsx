import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Cpu,
  Folder,
  FolderOpen,
  Gauge,
  HardDrive,
  Pickaxe,
  Play,
  Power,
  RefreshCw,
  Search,
  Server,
  Square,
  Trash2,
  Workflow
} from "lucide-react";
import {
  BackendClient,
  BackendConfig,
  DomPipelineStatus,
  HealthResponse,
  RasterProbe,
  SystemDevices
} from "./lib/api";
import { lastPathPart, formatPercent, formatDuration, timeStamp } from "./lib/format";
import { Button } from "./components/Button";
import { Field, TextInput, inputClass } from "./components/Field";
import { Card } from "./components/Card";
import { Badge } from "./components/Badge";
import { Toggle } from "./components/Toggle";
import { ProgressBar } from "./components/ProgressBar";
import { Stepper, StepItem, StepState } from "./components/Stepper";
import { ThemeSwitch } from "./components/ThemeSwitch";
import { AnimatedNumber } from "./components/AnimatedNumber";
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

  const [domProbe, setDomProbe] = useState<RasterProbe | null>(null);
  const [domProbeError, setDomProbeError] = useState<string | null>(null);
  const [domProbing, setDomProbing] = useState(false);
  const [epsgOverride, setEpsgOverride] = useState<string>("");

  const [devices, setDevices] = useState<SystemDevices | null>(null);

  const client = useMemo(() => (backend ? new BackendClient(backend) : null), [backend]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
    } catch {
      /* ignore */
    }
  }, [form]);

  useEffect(() => {
    void window.railCurve.backendConfig().then((config) => {
      setBackend(config);
      pushLog("info", `本地后端：${config.baseUrl}`);
    });
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

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

  const formIsValid =
    form.domPath.trim().length > 0 &&
    form.modelPath.trim().length > 0 &&
    form.outputDir.trim().length > 0;

  const effectiveEpsg: number | null = useMemo(() => {
    const overrideNum = parseInt(epsgOverride, 10);
    if (Number.isFinite(overrideNum) && overrideNum > 0) return overrideNum;
    if (domProbe?.epsg) return domProbe.epsg;
    return null;
  }, [domProbe, epsgOverride]);

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
    <div className="flex flex-col h-full bg-[var(--color-canvas)] relative">
      <TopBar
        backend={backend}
        health={health}
        pipelineRunning={pipelineRunning}
        themeMode={themeMode}
        onThemeChange={setThemeMode}
        devices={devices}
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

        <motion.main
          className="flex-1 min-w-0 flex flex-col gap-4 p-4 overflow-auto"
          initial="hidden"
          animate="show"
          variants={{
            hidden: {},
            show: { transition: { staggerChildren: 0.06, delayChildren: 0.05 } }
          }}
        >
          <FadeIn>
            <PipelineMonitor status={pipelineStatus} startedAt={pipelineStartTs} now={now} />
          </FadeIn>

          <div className="grid grid-cols-1 xl:grid-cols-[3fr_2fr] gap-4 flex-1 min-h-0">
            <FadeIn className="min-h-0">
              <LogPanel logs={logs} onClear={() => setLogs([])} />
            </FadeIn>
            <FadeIn className="min-h-0">
              <OutputsPanel status={pipelineStatus} />
            </FadeIn>
          </div>
        </motion.main>
      </div>
    </div>
  );
}

function FadeIn({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div
      className={className}
      variants={{
        hidden: { opacity: 0, y: 8 },
        show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] } }
      }}
    >
      {children}
    </motion.div>
  );
}

/* ---------------------------------- Top bar --------------------------------- */

function TopBar({
  backend,
  health,
  pipelineRunning,
  themeMode,
  onThemeChange,
  devices
}: {
  backend: BackendConfig | null;
  health: HealthResponse | null;
  pipelineRunning: boolean;
  themeMode: "system" | "light" | "dark";
  onThemeChange: (next: "system" | "light" | "dark") => void;
  devices: SystemDevices | null;
}) {
  return (
    <header className="relative flex items-center justify-between gap-4 h-14 px-4 border-b border-[var(--color-border)] bg-[var(--color-surface)] z-10">
      <div className="flex items-center gap-3 min-w-0">
        <Logo />
        <div className="flex flex-col min-w-0">
          <span className="text-[13px] font-semibold tracking-tight uppercase">
            Rail Curve Extractor
          </span>
          <span className="text-[10px] text-[var(--color-text-dim)] truncate font-mono uppercase tracking-wider">
            DOM → SEMSEG → POST → 3D CENTERLINE
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {devices ? <HardwarePill devices={devices} /> : null}

        <AnimatePresence>
          {pipelineRunning ? (
            <motion.div
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 8 }}
            >
              <Badge tone="info" dot pulse>
                PIPELINE RUNNING
              </Badge>
            </motion.div>
          ) : null}
        </AnimatePresence>

        <Badge tone={health?.ok ? "success" : "warn"} dot pulse={!!health?.ok}>
          {health?.ok ? `BACKEND · PY ${health.python}` : "WAITING"}
        </Badge>

        <span className="text-[10px] font-mono text-[var(--color-text-dim)] tabular-nums">
          {backend?.baseUrl ?? "—"}
        </span>

        <div className="w-px h-5 bg-[var(--color-border)]" />

        <ThemeSwitch mode={themeMode} onChange={onThemeChange} />
      </div>
    </header>
  );
}

function HardwarePill({ devices }: { devices: SystemDevices }) {
  const gpu = devices.cuda.gpus[0];
  const cudaUsable = devices.cuda.available && devices.torch.cuda_runtime_available;
  return (
    <div className="hidden md:flex items-center gap-3 px-3 h-7 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)]">
      <div className="flex items-center gap-1.5 text-[10px] font-mono">
        <Cpu size={11} className="text-[var(--color-text-muted)]" />
        <span className="text-[var(--color-text-muted)]">
          {devices.cpu.logical_cores ?? "?"}T
        </span>
      </div>
      <div className="w-px h-3 bg-[var(--color-border)]" />
      <div className="flex items-center gap-1.5 text-[10px] font-mono">
        <HardDrive size={11} className={cudaUsable ? "text-[var(--color-accent)]" : "text-[var(--color-text-dim)]"} />
        <span className={cudaUsable ? "text-[var(--color-accent)]" : "text-[var(--color-text-dim)]"}>
          {gpu?.memory_total_mib ? `${(gpu.memory_total_mib / 1024).toFixed(0)}GB` : "—"}
        </span>
      </div>
    </div>
  );
}

function Logo() {
  return (
    <div className="relative flex items-center justify-center w-8 h-8 rounded-md bg-[var(--color-accent-soft)] text-[var(--color-accent)] border border-[var(--color-accent)]/30">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
        <path d="M5 21 12 3l7 18" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M8.5 13h7" strokeLinecap="round" />
      </svg>
      <div
        className="absolute inset-0 rounded-md"
        style={{
          boxShadow: "0 0 14px var(--color-scan-glow)",
          opacity: 0.35
        }}
      />
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
    <aside className="w-[400px] shrink-0 border-r border-[var(--color-border)] bg-[var(--color-surface)] flex flex-col relative">
      <motion.div
        className="flex-1 min-h-0 overflow-auto p-4 flex flex-col gap-4"
        initial="hidden"
        animate="show"
        variants={{
          hidden: {},
          show: { transition: { staggerChildren: 0.04, delayChildren: 0.08 } }
        }}
      >
        <FadeIn>
          <SectionHeader icon={<Folder size={13} />} title="数据源" sub="必填三项 · 其余可选" />
        </FadeIn>

        <FadeIn>
          <PathPicker
            label="DOM 影像"
            required
            value={form.domPath}
            placeholder="dom.tif / dom.jpg"
            onChange={(value) => updateForm("domPath", value)}
            onPick={pickers.dom}
            disabled={inputsDisabled}
          />
        </FadeIn>

        <FadeIn>
          <CrsStatus
            domPath={form.domPath}
            probe={domProbe}
            probing={domProbing}
            error={domProbeError}
            epsgOverride={epsgOverride}
            setEpsgOverride={setEpsgOverride}
            disabled={inputsDisabled}
          />
        </FadeIn>

        <FadeIn>
          <PathPicker
            label="DeepLab 权重"
            required
            value={form.modelPath}
            placeholder="rail_semantic_deeplab_resnet50.pt"
            onChange={(value) => updateForm("modelPath", value)}
            onPick={pickers.model}
            disabled={inputsDisabled}
          />
        </FadeIn>

        <FadeIn>
          <PathPicker
            label="输出目录"
            required
            value={form.outputDir}
            placeholder="例：D:\output\dom_centerline_v1"
            onChange={(value) => updateForm("outputDir", value)}
            onPick={pickers.output}
            disabled={inputsDisabled}
          />
        </FadeIn>

        <FadeIn>
          <SectionHeader
            icon={<HardDrive size={13} />}
            title="3D 高程"
            sub="可选 · 缺省只产 2D 中心线"
          />
        </FadeIn>

        <FadeIn>
          <PathPicker
            label="DSM 栅格"
            value={form.dsmPath}
            placeholder="dsm.tif"
            onChange={(value) => updateForm("dsmPath", value)}
            onPick={pickers.dsm}
            disabled={inputsDisabled}
          />
        </FadeIn>

        <FadeIn>
          <PathPicker
            label="LAS / LAZ 目录"
            value={form.lasDir}
            placeholder="点云目录"
            onChange={(value) => updateForm("lasDir", value)}
            onPick={pickers.lasDir}
            disabled={inputsDisabled}
          />
        </FadeIn>

        <FadeIn>
          <SectionHeader icon={<Gauge size={13} />} title="运行参数" />
        </FadeIn>

        <FadeIn>
          <Field label="推理设备">
            <DevicePicker
              devices={devices}
              value={form.device}
              onChange={(value) => updateForm("device", value)}
              disabled={inputsDisabled}
            />
          </Field>
        </FadeIn>

        <FadeIn>
          <div className="grid grid-cols-2 gap-3">
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
        </FadeIn>

        <FadeIn>
          <Toggle
            checked={form.force}
            onChange={(value) => updateForm("force", value)}
            label="强制重跑"
            hint="忽略已有阶段产物，全部重新计算"
            disabled={inputsDisabled}
          />
        </FadeIn>
      </motion.div>

      <div className="border-t border-[var(--color-border)] p-3 bg-[var(--color-surface)]">
        {pipelineRunning ? (
          <Button variant="danger" className="w-full h-10" onClick={onStop} loading={busy} icon={<Square size={14} fill="currentColor" />}>
            停止流水线
          </Button>
        ) : (
          <Button
            variant="primary"
            className="w-full h-10"
            disabled={!canStart}
            loading={busy}
            onClick={onStart}
            icon={<Play size={14} fill="currentColor" />}
          >
            {busy ? "提交中" : "开始流水线"}
          </Button>
        )}
      </div>
    </aside>
  );
}

function SectionHeader({
  icon,
  title,
  sub
}: {
  icon: React.ReactNode;
  title: string;
  sub?: string;
}) {
  return (
    <div className="flex items-baseline gap-2 -mb-1">
      <span className="text-[var(--color-accent)]">{icon}</span>
      <h3 className="text-[12px] font-semibold tracking-[0.08em] text-[var(--color-text)] uppercase">
        {title}
      </h3>
      {sub ? (
        <span className="text-[10px] text-[var(--color-text-dim)] uppercase tracking-wider ml-auto">
          {sub}
        </span>
      ) : null}
    </div>
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
        <Button
          onClick={() => void onPick()}
          disabled={disabled}
          variant="secondary"
          size="md"
          icon={<FolderOpen size={13} />}
        >
          选择
        </Button>
      </div>
    </Field>
  );
}

/* ------------------------------- Device picker ------------------------------ */

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
  const cudaUsable = !!devices?.cuda.available && !!devices?.torch.cuda_runtime_available;
  const gpuName =
    devices?.cuda.gpus[0]?.name ?? (devices?.cuda.available ? "GPU" : "未检测到 NVIDIA GPU");
  const gpuMemoryGb = devices?.cuda.gpus[0]?.memory_total_mib
    ? `${(devices.cuda.gpus[0].memory_total_mib / 1024).toFixed(0)} GB`
    : null;
  const gpuDriver = devices?.cuda.gpus[0]?.driver_version;
  const cpuName = devices?.cpu.name ?? "CPU";
  const cpuCores =
    devices?.cpu.physical_cores && devices?.cpu.logical_cores
      ? `${devices.cpu.physical_cores}C / ${devices.cpu.logical_cores}T`
      : devices?.cpu.logical_cores
        ? `${devices.cpu.logical_cores}T`
        : null;

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
          icon={<HardDrive size={14} />}
          tag="CUDA"
          name={gpuName}
          meta={gpuMemoryGb}
          subMeta={gpuDriver ? `DRV ${gpuDriver}` : null}
          breathing={cudaUsable && value !== "cuda"}
          onClick={() => cudaUsable && onChange("cuda")}
        />
        <DeviceCard
          active={value === "cpu"}
          disabled={disabled}
          icon={<Cpu size={14} />}
          tag="CPU"
          name={cpuName}
          meta={cpuCores}
          subMeta={devices?.cpu.arch ?? null}
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
  tag,
  name,
  meta,
  subMeta,
  breathing,
  onClick
}: {
  active: boolean;
  disabled?: boolean;
  icon: React.ReactNode;
  tag: string;
  name: string;
  meta: string | null;
  subMeta: string | null;
  breathing?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "group relative flex flex-col gap-1.5 p-3 pl-4 rounded-md border text-left transition-colors min-w-0 overflow-hidden",
        active
          ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
          : "border-[var(--color-border)] bg-[var(--color-surface-2)] hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-3)]",
        disabled && "opacity-45 cursor-not-allowed hover:border-[var(--color-border)] hover:bg-[var(--color-surface-2)]"
      )}
    >
      {/* Accent bar slides in on selection */}
      <AnimatePresence>
        {active ? (
          <motion.span
            layoutId="device-accent"
            className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-[var(--color-accent-strong)]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ type: "spring", stiffness: 320, damping: 30 }}
          />
        ) : null}
      </AnimatePresence>

      {/* Idle breathing aura on usable-but-unselected GPU */}
      {breathing && !disabled ? (
        <span
          className="pointer-events-none absolute -top-4 -right-4 w-16 h-16 rounded-full bg-[var(--color-accent)]/20 blur-xl"
          style={{ animation: "breathe 3s ease-in-out infinite" }}
        />
      ) : null}

      <div className="flex items-center gap-2">
        <span
          className={clsx(
            "shrink-0 transition-colors",
            active ? "text-[var(--color-accent)]" : "text-[var(--color-text-muted)]"
          )}
        >
          {icon}
        </span>
        <span
          className={clsx(
            "text-[11px] font-mono font-bold tracking-wider uppercase",
            active ? "text-[var(--color-accent)]" : "text-[var(--color-text)]"
          )}
        >
          {tag}
        </span>
      </div>
      <div className="text-[12px] text-[var(--color-text)] leading-tight truncate" title={name}>
        {name}
      </div>
      <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
        <span className="text-[var(--color-text-muted)]">{meta ?? "—"}</span>
        {subMeta ? <span className="text-[var(--color-text-dim)]">{subMeta}</span> : null}
      </div>
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
          安装 GPU 版后才能用 CUDA 推理：从 pytorch.org 选 CUDA 12.x 的轮子。
        </span>
      </Hint>
    );
  }
  if (!cuda.available) {
    return (
      <Hint tone="info">
        未检测到 NVIDIA 显卡（或驱动未装），将使用 CPU 推理。
        <span className="block mt-0.5 text-[var(--color-text-dim)]">
          PyTorch CUDA 仅支持 NVIDIA。AMD/Intel 显卡在 Windows 上无法直接加速；如需 GPU 加速建议换 NVIDIA 卡或在 Linux 上配置 ROCm。
        </span>
      </Hint>
    );
  }
  return (
    <Hint tone="success">
      <span className="font-mono text-[10px]">
        CUDA {torch.cuda_build} · TORCH {torch.version}
      </span>
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

/* ----------------------------------- CRS ----------------------------------- */

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
  let label = "CRS";

  if (probing) {
    tone = "info";
    label = "PROBING";
    body = <span className="text-[var(--color-text-muted)]">正在读取 DOM 自带坐标系…</span>;
  } else if (error) {
    tone = "danger";
    label = "FAILED";
    body = (
      <span className="text-[var(--color-danger)] font-mono text-[11px] break-all">{error}</span>
    );
  } else if (probe?.epsg) {
    tone = "success";
    label = `EPSG:${probe.epsg}`;
    body = (
      <span className="text-[var(--color-text-muted)] truncate font-mono" title={probe.crs ?? ""}>
        {probe.crs ?? "—"}
      </span>
    );
  } else if (probe) {
    tone = "warn";
    label = "NO CRS";
    body = (
      <div className="flex flex-col gap-1.5">
        <span className="text-[var(--color-text-muted)]">
          这个 DOM 没有写入坐标系信息，需要手动指定。
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[var(--color-text-dim)]">EPSG</span>
          <TextInput
            mono
            value={epsgOverride}
            onChange={(event) => setEpsgOverride(event.currentTarget.value)}
            placeholder="32651"
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
    <motion.div
      layout
      className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 flex flex-col gap-1.5"
    >
      <div className="flex items-center justify-between gap-2">
        <Badge tone={tone} dot pulse={probing}>
          {label}
        </Badge>
        {probe?.width && probe?.height ? (
          <span className="text-[10px] font-mono text-[var(--color-text-dim)] tabular-nums">
            {probe.width}×{probe.height} · {probe.driver}
          </span>
        ) : null}
      </div>
      <div className="text-[11px] leading-relaxed">{body}</div>
    </motion.div>
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

  const idx = typeof status.stage_index === "number" ? status.stage_index : null;
  const total = typeof status.stage_count === "number" ? status.stage_count : null;
  const currentName = status.stage_name ?? "";
  let activeIdx = -1;
  if (currentName) {
    activeIdx = STAGE_BLUEPRINT.findIndex((s) => s.matchers.some((re) => re.test(currentName)));
  }
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
        <Badge tone={tone} dot pulse={status?.running}>
          {(status?.state ?? "idle").toUpperCase()}
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

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat
          label="进度"
          value={
            typeof status?.percent === "number" ? (
              <AnimatedNumber value={status.percent} fractionDigits={1} suffix="%" />
            ) : (
              "—"
            )
          }
        />
        <Stat
          label="阶段"
          value={
            status?.stage_index && status?.stage_count
              ? `${status.stage_index}/${status.stage_count}`
              : status?.stage_name ?? "—"
          }
        />
        <Stat label="耗时" value={status ? formatDuration(elapsed) : "—"} mono />
        <Stat label="进程" value={status?.pid ? `PID ${status.pid}` : "—"} mono />
      </div>

      <AnimatePresence>
        {status?.error ? (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="rounded-md border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 px-3 py-2 text-xs text-[var(--color-danger)] font-mono whitespace-pre-wrap break-all"
          >
            {status.error}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </Card>
  );
}

function Stat({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="relative rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 min-w-0 overflow-hidden">
      <div className="text-[10px] uppercase tracking-[0.1em] text-[var(--color-text-dim)] font-mono">
        {label}
      </div>
      <div
        className={clsx(
          "mt-1 text-sm font-semibold truncate text-[var(--color-text)] tabular-nums",
          mono ? "font-mono" : "font-mono"
        )}
      >
        {value}
      </div>
      {/* corner tick — industrial feel */}
      <div className="absolute top-0 right-0 w-2 h-2 border-r border-t border-[var(--color-border)]" />
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
      subtitle="最新事件位于顶部 · 仅前端事件流"
      actions={
        <>
          <div className="relative">
            <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-[var(--color-text-dim)]" />
            <input
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              placeholder="过滤"
              className={clsx(inputClass, "h-7 w-32 text-xs pl-7 pr-2 font-mono")}
            />
          </div>
          <Button size="sm" variant="ghost" onClick={handleCopy} disabled={filtered.length === 0}>
            复制
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onClear}
            disabled={logs.length === 0}
            icon={<Trash2 size={12} />}
          >
            清空
          </Button>
        </>
      }
      bodyClassName="p-0"
      className="min-h-0"
    >
      <div className="h-[320px] overflow-auto font-mono text-[11px] leading-relaxed">
        {filtered.length === 0 ? (
          <div className="h-full flex items-center justify-center text-[var(--color-text-dim)] text-xs">
            {logs.length === 0 ? "暂无日志" : "无匹配条目"}
          </div>
        ) : (
          <ul>
            <AnimatePresence initial={false}>
              {filtered.map((entry) => (
                <motion.li
                  key={entry.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                  className="flex gap-3 px-4 py-1 border-b border-[var(--color-border)]/40 last:border-b-0 hover:bg-[var(--color-surface-2)]"
                >
                  <span className="text-[var(--color-text-dim)] shrink-0 tabular-nums">
                    {entry.time}
                  </span>
                  <span className={clsx("shrink-0 w-12 text-[10px] uppercase tracking-wider font-bold", logLevelColor(entry.level))}>
                    {entry.level}
                  </span>
                  <span className="text-[var(--color-text)] break-all">{entry.text}</span>
                </motion.li>
              ))}
            </AnimatePresence>
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
          <Button size="sm" variant="secondary" onClick={() => void openFolder(outDir)} icon={<FolderOpen size={12} />}>
            打开输出目录
          </Button>
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
          <ul>
            {outDir ? (
              <OutputRow
                kind="DIR"
                label="out_dir"
                path={outDir}
                onReveal={() => void openFolder(outDir)}
                onCopy={() => void copy(outDir)}
              />
            ) : null}
            {entries.length === 0 ? (
              <li className="px-4 py-3 text-xs text-[var(--color-text-dim)] font-mono">
                {status.state === "running" || status.state === "starting"
                  ? "运行中，等待阶段输出…"
                  : "暂无具名产出文件"}
              </li>
            ) : (
              entries.map((entry) => (
                <OutputRow
                  key={entry.key}
                  kind="FILE"
                  label={entry.key}
                  path={entry.value}
                  onReveal={() => void reveal(entry.value)}
                  onCopy={() => void copy(entry.value)}
                />
              ))
            )}
            {summaryPath ? (
              <OutputRow
                kind="JSON"
                label="summary"
                path={summaryPath}
                onReveal={() => void reveal(summaryPath)}
                onCopy={() => void copy(summaryPath)}
              />
            ) : null}
            {logPath ? (
              <OutputRow
                kind="LOG"
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
    <li className="group flex items-center gap-3 px-4 py-2 border-b border-[var(--color-border)]/40 last:border-b-0 hover:bg-[var(--color-surface-2)] transition-colors">
      <span className="shrink-0 inline-flex items-center justify-center min-w-[42px] h-5 px-1.5 rounded text-[9px] font-mono font-bold tracking-wider bg-[var(--color-surface-3)] text-[var(--color-text-muted)] border border-[var(--color-border)]">
        {kind}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-xs font-medium text-[var(--color-text)] truncate">{label}</div>
        <div className="text-[10px] font-mono text-[var(--color-text-dim)] truncate" title={path}>
          {path}
        </div>
      </div>
      <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1 shrink-0">
        <Button size="sm" variant="ghost" onClick={onCopy}>
          复制
        </Button>
        <Button size="sm" variant="ghost" onClick={onReveal} icon={<FolderOpen size={11} />}>
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

void lastPathPart;
void formatPercent;
void Activity;
void Server;
void Workflow;
void Pickaxe;
void Power;
void RefreshCw;
