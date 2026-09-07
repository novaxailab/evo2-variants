import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Delta likelihood scores cluster around 1e-3 to 1e-5, so a plain toFixed(6)
 * renders most real results as "0.000000". Fall back to scientific notation
 * once a score is too small to show meaningfully in decimal form.
 */
export function formatDeltaScore(score: number) {
  if (score !== 0 && Math.abs(score) < 0.001) return score.toExponential(3);
  return score.toFixed(6);
}
