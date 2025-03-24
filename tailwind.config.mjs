import { fontFamily } from "tailwindcss/defaultTheme";

/** @type {import('tailwindcss').Config} */
const config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Light Theme Tokens
        "light-bg": "#F7F7F7",
        "light-primary": "#17BEBB",
        "light-secondary": "#706993",
        "light-tertiary": "#331E38",
        "light-text": "#331E38",

        // Dark Theme Tokens
        "dark-bg": "#191923",
        "dark-primary": "#7D31AF",
        "dark-secondary": "#B5BA72",
        "dark-tertiary": "#99907D",
        "dark-text": "#504B3F",

        // Semantic Tokens (Used in app)
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
