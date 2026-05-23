export {};

declare global {
  interface RailCurveBridge {
    backendConfig: () => Promise<{ baseUrl: string; token: string }>;
    openDomDialog: () => Promise<string | null>;
    openModelDialog: () => Promise<string | null>;
    openDsmDialog: () => Promise<string | null>;
    openLasDirectoryDialog: () => Promise<string | null>;
    selectOutputDirectory: () => Promise<string | null>;
    revealInExplorer: (target: string) => Promise<boolean>;
    openPath: (target: string) => Promise<string>;
    writeClipboard: (text: string) => Promise<boolean>;
  }

  interface Window {
    railCurve: RailCurveBridge;
  }
}

declare module "*.svg" {
  const content: string;
  export default content;
}
