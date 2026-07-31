"use client";

import {
  BookOpen,
  CloudDownload,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { useState } from "react";

import { useLanguage } from "./LanguageProvider";


interface KnowledgeItem {
  id: string;
  name: string;
  source: "official" | "local";
  updatedAt: string;
  size: string;
}


const MOCK_KNOWLEDGE: KnowledgeItem[] = [
  { id: "1", name: "通用翻译术语库", source: "official", updatedAt: "2026-07-20", size: "2.4 MB" },
  { id: "2", name: "动漫字幕常用表达", source: "official", updatedAt: "2026-07-15", size: "1.1 MB" },
  { id: "3", name: "技术文档翻译规范", source: "official", updatedAt: "2026-06-30", size: "860 KB" },
];


export function KnowledgeBase() {
  const { t } = useLanguage();
  const [items, setItems] = useState<KnowledgeItem[]>(MOCK_KNOWLEDGE);
  const [syncing, setSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState("");

  const syncFromCloud = async () => {
    setSyncing(true);
    setSyncMessage(t.knowledge.syncing);
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setSyncMessage(t.knowledge.syncComplete);
    setSyncing(false);
  };

  const uploadLocal = () => {
    const localCount = items.filter((i) => i.source === "local").length + 1;
    const newItem: KnowledgeItem = {
      id: String(Date.now()),
      name: `${t.knowledge.customName} ${localCount}`,
      source: "local",
      updatedAt: new Date().toISOString().slice(0, 10),
      size: "—",
    };
    setItems((prev) => [...prev, newItem]);
    setSyncMessage(t.knowledge.imported);
  };

  const removeItem = (id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">{t.knowledge.kicker}</p> */}
          <h1>{t.knowledge.title}</h1>
          <p>{t.knowledge.description}</p>
        </div>
      </header>

      <section className="knowledge-actions">
        <button
          type="button"
          className="button button-primary"
          disabled={syncing}
          onClick={() => void syncFromCloud()}
        >
          {syncing ? <RefreshCw size={14} className="spin" /> : <CloudDownload size={14} />}
          {t.knowledge.syncOfficial}
        </button>
        <button
          type="button"
          className="button button-secondary"
          onClick={uploadLocal}
        >
          <Upload size={14} />
          {t.knowledge.uploadLocal}
        </button>
        {syncMessage ? <span className="knowledge-message">{syncMessage}</span> : null}
      </section>

      {items.length ? (
        <section className="knowledge-list">
          {items.map((item) => (
            <article className="knowledge-row" key={item.id}>
              <span className="knowledge-icon">
                <BookOpen size={17} />
              </span>
              <div className="knowledge-copy">
                <strong>{item.name}</strong>
                <span>
                  {item.source === "official" ? t.knowledge.sourceOfficial : t.knowledge.sourceLocal} · {item.size} · {t.knowledge.updatedAt} {item.updatedAt}
                </span>
              </div>
              <div className="knowledge-side">
                <span className={`knowledge-tag is-${item.source}`}>
                  {item.source === "official" ? t.knowledge.sourceOfficial : t.knowledge.sourceLocal}
                </span>
                <button
                  type="button"
                  className="button button-danger-quiet button-compact"
                  onClick={() => removeItem(item.id)}
                >
                  <Trash2 size={14} /> {t.knowledge.delete}
                </button>
              </div>
            </article>
          ))}
        </section>
      ) : (
        <section className="empty-state">
          <BookOpen size={24} />
          <h2>{t.knowledge.emptyTitle}</h2>
          <p>{t.knowledge.emptyDescription}</p>
        </section>
      )}
    </div>
  );
}