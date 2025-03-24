import "../styles/globals.css";
import { ThemeProvider } from "next-themes";
import { inter, firaCode, lato } from "./fonts";

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
    <html
      lang="en"
      className={`${inter.variable} ${firaCode.variable} ${lato.variable}`}
      suppressHydrationWarning
    >
      <body className="bg-light-bg text-light-text dark:bg-dark-bg dark:text-dark-text font-body transition-colors">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
