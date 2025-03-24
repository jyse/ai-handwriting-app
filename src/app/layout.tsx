import "../styles/globals.css";
import { ThemeProvider } from "next-themes";

export const metadata = {
  title: "AI Handwriting App",
  description: "Create fonts with your handwriting using AI."
};
export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="bg-bg text-text font-body">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
