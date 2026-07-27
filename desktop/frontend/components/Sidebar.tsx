"use client";

import {
  Boxes,
  Clock3,
  Plus,
  Settings2,
  Sparkles,
  Download,
} from "lucide-react";

import type {
  CapabilityState,
  ResourceInstallSnapshot,
  Route,
} from "@/lib/types";


const navigation: Array<{
  route: Route;
  label: string;
  icon: typeof Plus;
}> = [
  { route: "new-task", label: "新建任务", icon: Plus },
  { route: "history", label: "任务记录", icon: Clock3 },
  { route: "resources", label: "运行资源", icon: Boxes },
  { route: "settings", label: "设置", icon: Settings2 },
];


interface SidebarProps {
  route: Route;
  capabilities: CapabilityState;
  resourceInstalls: ResourceInstallSnapshot[];
  onNavigate: (route: Route) => void;
}


export function Sidebar({
  route,
  capabilities,
  resourceInstalls,
  onNavigate,
}: SidebarProps) {
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
      <nav className="sidebar-nav" aria-label="主导航">
        <p className="sidebar-section-label">工作区</p>
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
              <strong>资源正在后台处理</strong>
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
              {capabilities.translation ? "翻译已就绪" : "本地识别可用"}
            </strong>
            <span>
              {capabilities.translation
                ? "Gemini 已连接"
                : "翻译功能可选配置"}
            </span>
          </div>
        </div>
        <div className="sidebar-version">
          <Sparkles size={13} />
          <span>FineSub Desktop 0.2.7</span>
        </div>
      </div>
    </aside>
  );
}
