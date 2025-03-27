// app/layout.tsx or a component where context is available
"use client";
import { ThemeProvider, useTheme } from "next-themes";
import { Toaster } from "react-hot-toast";

export default function ThemedToaster() {

  const { resolvedTheme } = useTheme();

  const isDark = resolvedTheme === "dark";

  return (
    <Toaster
      position="top-right"
      toastOptions={{
        style: {
          background: isDark ? "#1e1e2f" : "#fff",
          color: isDark ? "#fff" : "#1e1e2f",
          fontSize: "0.875rem",
          borderRadius: "0.5rem",
        },
        success: {
          iconTheme: {
            primary: "#7d31af",
            secondary: isDark ? "#fff" : "#000",
          },
        },
      }}
    />
  );
}
