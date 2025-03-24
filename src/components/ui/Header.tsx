"use client";
import React from "react";
import ThemeToggle from "./ThemeToggle";

const Header = () => {
  return (
    <header className=" bg-dark text-primary w-full px-6 py-4 flex items-center justify-between border-b border-secondary">
      <h1 className="text-xl font-heading">Konnichiwa AI Handwriting App</h1>
      <ThemeToggle />
    </header>
  );
};

export default Header;
