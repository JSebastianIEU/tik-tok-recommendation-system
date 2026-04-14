import React from "react";
import ReactDOM from "react-dom/client";
import { AuthProvider } from "./app/AuthContext";
import { App } from "./app/App";
import "./styles/globals.css";
import "./styles/responsive.css";
import "./keyboard";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>
);
