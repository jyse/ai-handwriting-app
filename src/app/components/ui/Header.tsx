"use client";
import React from "react";
import ThemeToggle from "../ui/ThemeToggle";

const Header = () => {
  return (
    <header className="w-full flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-zinc-800">
      <h1 className="text-xl font-bold">AI Handwriting App</h1>

      <div className="flex items-center space-x-4">
        <ThemeToggle />
      </div>
    </header>
  );
};

export default Header;
