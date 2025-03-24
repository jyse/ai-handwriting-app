// src/components/ui/PageWrapper.tsx
import React from "react";

type Props = {
  children: React.ReactNode;
};

export default function PageWrapper({ children }: Props) {
  return (
    <main className="max-w-4xl mx-auto px-6 py-12 text-left">{children}</main>
  );
}
