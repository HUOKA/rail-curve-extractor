export {};

declare global {
  interface RailCurveBridge {
    backendConfig: () => Promise<{ baseUrl: string; token: string }>;
    openPointCloudDialog: () => Promise<string | null>;
    openPointCloudFolderDialog: () => Promise<string | null>;
    openDomDialog: () => Promise<string | null>;
    openModelDialog: () => Promise<string | null>;
    openDsmDialog: () => Promise<string | null>;
    openLasDirectoryDialog: () => Promise<string | null>;
    selectOutputDirectory: () => Promise<string | null>;
  }

  interface Window {
    railCurve: RailCurveBridge;
  }
}
