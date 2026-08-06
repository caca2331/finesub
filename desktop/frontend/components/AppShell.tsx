"use client";

import type { ReactNode } from "react";

import type { AppState } from "@/lib/state";
import type { DesktopApi, Route } from "@/lib/types";

import { Sidebar } from "./Sidebar";
import { TitleBar } from "./TitleBar";


interface AppShellProps {
  state: AppState;
  api: DesktopApi;
  onNavigate: (route: Route) => void;
  children: ReactNode;
}


export function AppShell({
  state,
  api,
  onNavigate,
  children,
}: AppShellProps) {
  return (
    <div className="app-shell">
      <TitleBar api={api} />
      <Sidebar
        route={state.route}
        capabilities={state.capabilities}
        resourceInstalls={state.resourceInstalls}
        appVersion={state.appVersion}
        onNavigate={onNavigate}
      />
      <section className="workspace">
        <div className="workspace-view" key={state.route}>
          {children}
        </div>
      </section>
    </div>
  );
}
