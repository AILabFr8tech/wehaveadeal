"use client";

import { useRouter } from "next/navigation";

export function RoundPicker({ files, current }: { files: string[]; current: string }) {
  const router = useRouter();
  if (files.length <= 1) return null;
  return (
    <label className="font-mono text-xs text-ink-muted flex items-center gap-sm">
      round
      <select
        className="bg-surface-light dark:bg-surface-dark border border-line-light dark:border-line-dark rounded px-sm py-xs text-ink-primary-light dark:text-ink-primary-dark font-mono text-xs"
        value={current}
        onChange={(e) => router.push(`/round/${e.target.value}`)}
      >
        {files.map((f) => (
          <option key={f} value={f}>
            {f}
          </option>
        ))}
      </select>
    </label>
  );
}
