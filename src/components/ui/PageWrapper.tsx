// src/components/ui/PageWrapper.tsx
import React from "react";

type Props = {
  children: React.ReactNode;
};

export default function PageWrapper({ children }: Props) {
  return (
    <main className="min-h-screen flex flex-col items-center py-10 px-10">
      <main className="w-full max-w-4xl flex flex-col gap-6">{children}</main>
    </main>
  );
}
