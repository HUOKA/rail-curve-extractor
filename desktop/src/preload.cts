const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("railCurve", {
  backendConfig: () => ipcRenderer.invoke("backend-config"),
  openDomDialog: () => ipcRenderer.invoke("dialog:open-dom"),
  openModelDialog: () => ipcRenderer.invoke("dialog:open-model"),
  openDsmDialog: () => ipcRenderer.invoke("dialog:open-dsm"),
  openLasDirectoryDialog: () => ipcRenderer.invoke("dialog:open-las-dir"),
  selectOutputDirectory: () => ipcRenderer.invoke("dialog:select-output-dir"),
  revealInExplorer: (target: string) => ipcRenderer.invoke("shell:reveal", target),
  openPath: (target: string) => ipcRenderer.invoke("shell:open-path", target),
  writeClipboard: (text: string) => ipcRenderer.invoke("clipboard:write", text)
});
