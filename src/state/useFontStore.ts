import { create } from "zustand";

interface FontState {
  thickness: number;
  spacing: number;
  slant: number;
  setThickness: (value: number) => void;
  setSpacing: (value: number) => void;
  setSlant: (value: number) => void;
}

export const useFontStore = create<FontState>((set) => ({
  thickness: 1,
  spacing: 1,
  slant: 0,
  setThickness: (value) => set({ thickness: value }),
  setSpacing: (value) => set({ spacing: value }),
  setSlant: (value) => set({ slant: value })
}));
