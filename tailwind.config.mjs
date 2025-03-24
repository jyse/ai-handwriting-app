import { fontFamily } from "tailwindcss/defaultTheme";

/** @type {import('tailwindcss').Config} */
const config = {
  // Enable dark mode based on the `class` on <html>
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Semantic design tokens using CSS variables
      // These names (bg, text, primary...) are used as Tailwind classnames like `bg-bg`, `text-primary`
      colors: {
        bg: "var(--color-bg)",
        text: "var(--color-text)",
        primary: "var(--color-primary)",
        secondary: "var(--color-secondary)",
        tertiary: "var(--color-tertiary)"
      },
      fontFamily: {
        mono: ["var(--font-fira-code)", ...fontFamily.mono],
        heading: ["var(--font-lato)", ...fontFamily.sans],
        body: ["var(--font-inter)", ...fontFamily.sans]
      },
      spacing: {
        layout: "1.5rem",
        gutter: "2rem"
      }
    }
  },
  plugins: []
};

export default config;
