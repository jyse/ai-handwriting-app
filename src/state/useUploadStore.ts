import { create } from "zustand";

interface UploadStore {
  file: File | null;
  previewUrl: string | null;
  setFile: (file: File, previewUrl: string) => void;
  reset: () => void;
}

export const useUploadStore = create<UploadStore>((set) => ({
  file: null,
  previewUrl: null,
  setFile: (file, previewUrl) => set({ file, previewUrl }),
  reset: () => set({ file: null, previewUrl: null })
}));
