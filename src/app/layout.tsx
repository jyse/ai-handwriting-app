import "../styles/globals.css";
import { ThemeProvider } from "next-themes";
import Header from "../components/ui/Header";
import Footer from "../components/ui/Footer";

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
    <html lang="en" suppressHydrationWarning>
      <body className="bg-bg text-text font-body">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="min-h-screen flex flex-col">
            <Header />
            <main className="min-h-screen">{children}</main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
