export {};

declare global {
  interface RailCurveBridge {
    backendConfig: () => Promise<{ baseUrl: string; token: string }>;
    openPointCloudDialog: () => Promise<string | null>;
    openPointCloudFolderDialog: () => Promise<string | null>;
    selectOutputDirectory: () => Promise<string | null>;
  }

  interface Window {
    railCurve: RailCurveBridge;
  }
}
