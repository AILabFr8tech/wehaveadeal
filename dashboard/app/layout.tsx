import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "We Have a Deal — Arena",
  description: "Negotiation tournament arena: market, matching and scoring at a glance.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-plane-light dark:bg-plane-dark text-ink-primary-light dark:text-ink-primary-dark font-sans antialiased min-h-screen">
        <div className="border-b border-line-light dark:border-line-dark">
          <div className="max-w-[1400px] mx-auto px-lg py-md flex items-center justify-between flex-wrap gap-sm">
            <Link href="/" className="flex items-baseline">
              <span className="font-mono text-sm tracking-tight text-ink-muted">WE HAVE A DEAL</span>
              <span className="mx-sm text-ink-muted">/</span>
              <span className="font-mono text-sm text-ink-secondary-light dark:text-ink-secondary-dark">
                test bench
              </span>
            </Link>
            <nav className="flex items-center gap-md font-mono text-xs">
              <Link href="/" className="text-ink-secondary-light dark:text-ink-secondary-dark hover:underline">
                history
              </Link>
              <Link
                href="/live"
                className="px-sm py-xs rounded bg-status-critical/10 text-status-critical border border-status-critical/30 hover:bg-status-critical/20"
              >
                ● live negotiation
              </Link>
              <span className="text-ink-muted hidden sm:inline">9 reference engines + your agents</span>
            </nav>
          </div>
        </div>
        <div className="max-w-[1400px] mx-auto px-lg py-lg">{children}</div>
      </body>
    </html>
  );
}
