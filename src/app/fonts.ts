import { Fira_Code } from "next/font/google";
import { Lato } from "next/font/google";
import { Inter } from "next/font/google";

export const firaCode = Fira_Code({
  subsets: ["latin"],
  variable: "--font-fira-code",
  weight: ["500"],
  display: "swap"
});

export const lato = Lato({
  subsets: ["latin"],
  variable: "--font-lato",
  weight: ["700"],
  display: "swap"
});

export const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500"],
  display: "swap"
});
