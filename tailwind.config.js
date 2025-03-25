/** @type {import('tailwindcss').Config} */
const { fontFamily } = require("tailwindcss/defaultTheme");

module.exports = {
  // ✅ Enables class-based dark mode (required for next-themes)
  darkMode: "class",

  // ✅ Scans all relevant folders for Tailwind class usage
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
    "./src/design-system/**/*.{js,ts,jsx,tsx}",
    "./src/styles/**/*.{js,ts,jsx,tsx}"
  ],

  theme: {
    extend: {
      colors: {
        // Semantic tokens using CSS variables (defined in globals.css)
        bg: "var(--color-bg)",
        text: "var(--color-text)",
        primary: "var(--color-primary)",
        secondary: "var(--color-secondary)",
        tertiary: "var(--color-tertiary)",
        quartiary: "var(--color-quartiary)"
      },
      fontFamily: {
        // Monospace font for headings
        mono: ["var(--font-fira-code)", ...fontFamily.mono],
        // Modern clean fonts for heading/body
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
