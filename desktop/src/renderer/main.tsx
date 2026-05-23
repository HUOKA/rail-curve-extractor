import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { bootstrapThemeBeforeMount } from "./lib/theme";
import "./styles.css";

bootstrapThemeBeforeMount();

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("#root not found");
}

createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
