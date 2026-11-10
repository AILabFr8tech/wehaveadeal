import type { Config } from "tailwindcss";

// Tokens lifted from the dataviz skill's validated reference palette
// (references/palette.md) -- categorical hues are CVD-validated and must be
// assigned in this fixed order, never cycled or reassigned by rank.
const config: Config = {
  darkMode: "media",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        plane: { light: "#f9f9f7", dark: "#0d0d0d" },
        surface: { light: "#fcfcfb", dark: "#1a1a19" },
        ink: {
          primary: { light: "#0b0b0b", dark: "#ffffff" },
          secondary: { light: "#52514e", dark: "#c3c2b7" },
          muted: "#898781",
        },
        line: { light: "#e1e0d9", dark: "#2c2c2a" },
        baseline: { light: "#c3c2b7", dark: "#383835" },
        series: {
          1: { light: "#2a78d6", dark: "#3987e5" }, // blue    -- CarrierR
          2: { light: "#eb6834", dark: "#d95926" }, // orange  -- CarrierRV
          3: { light: "#1baf7a", dark: "#199e70" }, // aqua    -- ShipperR (sale price)
          4: { light: "#eda100", dark: "#c98500" }, // yellow  -- goods cost
        },
        status: {
          good: "#0ca30c",
          warning: "#fab219",
          serious: "#ec835a",
          critical: "#d03b3b",
        },
      },
      fontFamily: {
        sans: [
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      spacing: {
        xs: "4px",
        sm: "8px",
        md: "16px",
        lg: "24px",
        xl: "32px",
        "2xl": "48px",
      },
    },
  },
  plugins: [],
};

export default config;
