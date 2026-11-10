// Locale is pinned to "en-US" (not `undefined`) everywhere in this file and
// its callers: `undefined` defers to the runtime's default locale, which can
// differ between the Node server and the browser and causes a hydration
// mismatch on every server-rendered number.
export function money(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function num(n: number, digits = 0): string {
  return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
