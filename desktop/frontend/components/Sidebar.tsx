"use client";

import {
  Boxes,
  Clock3,
  Plus,
  Settings2,
  Download,
} from "lucide-react";

import type {
  CapabilityState,
  ResourceInstallSnapshot,
  Route,
} from "@/lib/types";

import { useLanguage } from "./LanguageProvider";


interface SidebarProps {
  route: Route;
  capabilities: CapabilityState;
  resourceInstalls: ResourceInstallSnapshot[];
  appVersion: string;
  onNavigate: (route: Route) => void;
}


export function Sidebar({
  route,
  capabilities,
  resourceInstalls,
  appVersion,
  onNavigate,
}: SidebarProps) {
  const { t } = useLanguage();
  
  const navigation: Array<{
    route: Route;
    label: string;
    icon: typeof Plus;
  }> = [
    { route: "new-task", label: t.sidebar.newTask, icon: Plus },
    { route: "history", label: t.sidebar.history, icon: Clock3 },
    { route: "resources", label: t.sidebar.resources, icon: Boxes },
    { route: "settings", label: t.sidebar.settings, icon: Settings2 },
  ];
  
  const activeInstall = resourceInstalls.find(
    (install) => install.state === "queued" || install.state === "running",
  );
  const activePercent =
    activeInstall && activeInstall.total > 0
      ? Math.min(
          100,
          Math.round(
            (activeInstall.downloaded / activeInstall.total) * 100,
          ),
        )
      : null;
  return (
    <aside className="sidebar">
      <nav className="sidebar-nav" aria-label={t.sidebar.navigationAria}>
        {navigation.map((item) => {
          const Icon = item.icon;
          const active = route === item.route;
          return (
            <button
              type="button"
              key={item.route}
              className={`nav-item${active ? " is-active" : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => onNavigate(item.route)}
            >
              <Icon size={16} strokeWidth={1.8} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-meta">
        {activeInstall ? (
          <button
            type="button"
            className="sidebar-download-status"
            onClick={() => onNavigate("resources")}
          >
            <Download size={14} />
            <span>
              <strong>{t.sidebar.resourceProcessing}</strong>
              <small>
                {activePercent === null
                  ? activeInstall.message
                  : `${activePercent}% · ${activeInstall.message}`}
              </small>
            </span>
          </button>
        ) : null}
        <div className="sidebar-capability">
          <span
            className={`status-dot ${
              capabilities.translation ? "is-ready" : "is-neutral"
            }`}
          />
          <div>
            <strong>
              {capabilities.translation ? t.sidebar.translationReady : t.sidebar.localOnly}
            </strong>
            <span>
              {capabilities.translation
                ? t.sidebar.geminiConnected
                : t.sidebar.translationOptional}
            </span>
          </div>
        </div>
        <div className="sidebar-version">
          <span>FineSub Desktop v{appVersion}</span>
        </div>
      </div>
    </aside>
  );
}
