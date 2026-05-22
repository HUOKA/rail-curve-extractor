const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("railCurve", {
  backendConfig: () => ipcRenderer.invoke("backend-config"),
  openPointCloudDialog: () => ipcRenderer.invoke("dialog:open-point-cloud"),
  openPointCloudFolderDialog: () => ipcRenderer.invoke("dialog:open-point-cloud-folder"),
  selectOutputDirectory: () => ipcRenderer.invoke("dialog:select-output-dir")
});
