"use client";

import { LoaderCircle } from "lucide-react";

import { useLanguage } from "@/components/LanguageProvider";
import type { BridgeError } from "@/lib/types";

interface BootstrapScreenProps {
  error: BridgeError | null;
  onRetry: () => void;
}

export function BootstrapScreen({ error, onRetry }: BootstrapScreenProps) {
  const { t } = useLanguage();

  return (
    <main className="bootstrap-screen">
      <div className="bootstrap-brand">
        <img
          className="brand-icon"
          src="./icon.png"
          alt=""
          draggable={false}
        />
        <strong>{t.bootstrap.brand}</strong>
      </div>
      {error ? (
        <div className="bootstrap-error">
          <h1>{t.bootstrap.connectionError}</h1>
          <p>{error.message}</p>
          <button
            type="button"
            className="button button-primary"
            onClick={() => void onRetry()}
          >
            {t.bootstrap.reconnect}
          </button>
        </div>
      ) : (
        <div className="bootstrap-loading">
          <LoaderCircle size={20} className="spin" />
          <span>{t.bootstrap.loading}</span>
        </div>
      )}
    </main>
  );
}