/**
 * One definition of "ready", "required" and "installing".
 *
 * These questions used to be answered by five predicates that disagreed:
 * the start-task gate treated *every* not-ready resource as blocking, while
 * the environment panel and the resource list both filtered `optional` out.
 * That difference was the bug — `yt-dlp` has no system-tool finder by design,
 * so it is permanently "missing" on a healthy machine, and the gate therefore
 * refused to start any task until the user installed both on-demand tools.
 * The backend flags them optional precisely so that cannot happen.
 */

import type { ResourceInstallSnapshot, ResourceStatus } from "./types";

/** Resources a task genuinely cannot run without. */
export function requiredResources(resources: ResourceStatus[]): ResourceStatus[] {
  return resources.filter((resource) => !resource.optional);
}

/**
 * Whether a task can run on what is installed right now.
 *
 * Not `state === "ready"`: that also asserts "and it is the newest version",
 * which is a different claim. An outdated tool still works, and treating the
 * two alike turned every yt-dlp version bump into a wall in front of users who
 * had a perfectly good copy.
 */
export function isUsable(resource: ResourceStatus): boolean {
  return resource.state === "ready" || resource.state === "outdated";
}

/** The dependency that still blocks this resource, if any. */
export function unresolvedDependency(
  resource: ResourceStatus,
  resources: ResourceStatus[],
): string {
  if (!resource.blocked_by || isUsable(resource)) {
    return "";
  }
  const dependency = resources.find(
    (candidate) => candidate.id === resource.blocked_by,
  );
  // An absent dependency is not proof that it is ready. Keep the action safe
  // until the next bootstrap supplies a usable status for it.
  return dependency && isUsable(dependency) ? "" : resource.blocked_by;
}

/** Required resources that are not usable — i.e. what actually blocks a task. */
export function blockingResources(resources: ResourceStatus[]): ResourceStatus[] {
  return requiredResources(resources).filter(
    (resource) => !isUsable(resource),
  );
}

export function isEnvironmentReady(resources: ResourceStatus[]): boolean {
  return blockingResources(resources).length === 0;
}

/** An install that is under way, so the UI should keep polling it. */
export function isInstallActive(install: ResourceInstallSnapshot): boolean {
  return install.state === "queued" || install.state === "running";
}

export function hasActiveInstall(installs: ResourceInstallSnapshot[]): boolean {
  return installs.some(isInstallActive);
}
