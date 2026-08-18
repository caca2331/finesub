"use client";

import { saveTaskDefaults, taskDefaults } from "./preferences";
import type { GpuSnapshot } from "./types";

/** null = let CUDA choose (and what every single-GPU machine wants). */
export type ProcessingDevice = {
  device: "cuda" | "cpu";
  gpuIndex: number | null;
  /** What that index named when it was picked; "" when it does not apply. */
  gpuName: string;
};

export const AUTOMATIC: ProcessingDevice = {
  device: "cuda",
  gpuIndex: null,
  gpuName: "",
};

/**
 * Stored as the task fields it becomes (`device` / `gpu_index` / `gpu_name`)
 * rather than as a shape of its own: this is "what I picked last time" for a
 * task option, not an appearance preference.
 */
export function readProcessingDevice(): ProcessingDevice {
  const defaults = taskDefaults();
  if (defaults.device === "cpu") {
    return { device: "cpu", gpuIndex: null, gpuName: "" };
  }
  const index = defaults.gpu_index;
  if (!Number.isInteger(index) || (index as number) < 0) {
    return AUTOMATIC;
  }
  return {
    device: "cuda",
    gpuIndex: index as number,
    gpuName: typeof defaults.gpu_name === "string" ? defaults.gpu_name : "",
  };
}

export function writeProcessingDevice(choice: ProcessingDevice): void {
  if (choice.device === "cpu") {
    saveTaskDefaults({ device: "cpu", gpu_index: null, gpu_name: null });
    return;
  }
  if (choice.gpuIndex === null) {
    // Automatic: back to the code default, so the keys go away entirely.
    saveTaskDefaults({ device: null, gpu_index: null, gpu_name: null });
    return;
  }
  saveTaskDefaults({
    device: "cuda",
    gpu_index: choice.gpuIndex,
    gpu_name: choice.gpuName,
  });
}

/**
 * Whether a stored choice still names a card this machine has.
 *
 * Only meaningful once the probe has answered -- while it is scanning the
 * choice is left alone, because nothing may wait for it and a stale index is
 * better than second-guessing a machine we have not looked at yet. The backend
 * makes the same check again when a task starts, which is what covers retry
 * and resume.
 */
export function isStale(
  choice: ProcessingDevice,
  gpus: GpuSnapshot | undefined,
): boolean {
  if (choice.gpuIndex === null || gpus === undefined || gpus.state !== "ready") {
    return false;
  }
  const present = gpus.devices.find((device) => device.index === choice.gpuIndex);
  if (present === undefined) {
    return true;
  }
  // Same slot, different card. The backend falls back on this too, and the
  // settings page must not call that healthy while the task says otherwise.
  return choice.gpuName !== "" && present.name !== choice.gpuName;
}
