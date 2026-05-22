const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("railCurve", {
  backendConfig: () => ipcRenderer.invoke("backend-config"),
  openPointCloudDialog: () => ipcRenderer.invoke("dialog:open-point-cloud"),
  openPointCloudFolderDialog: () => ipcRenderer.invoke("dialog:open-point-cloud-folder"),
  openDomDialog: () => ipcRenderer.invoke("dialog:open-dom"),
  openModelDialog: () => ipcRenderer.invoke("dialog:open-model"),
  openDsmDialog: () => ipcRenderer.invoke("dialog:open-dsm"),
  openLasDirectoryDialog: () => ipcRenderer.invoke("dialog:open-las-dir"),
  selectOutputDirectory: () => ipcRenderer.invoke("dialog:select-output-dir")
});
