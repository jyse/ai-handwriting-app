import { create } from "zustand";

interface UploadStore {
  file: File | null;
  previewUrl: string | null;
  isProcessing: boolean;
  ocrResult: string[] | null;
  error: string | null;

  // Actions
  setFile: (file: File, previewUrl: string) => void;
  setProcessing: (bool: boolean) => void;
  setResult: (letters: string[]) => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useUploadStore = create<UploadStore>((set) => ({
  file: null,
  previewUrl: null,
  isProcessing: false,
  ocrResult: null,
  error: null,

  setFile: (file, previewUrl) => set({ file, previewUrl }),
  setProcessing: (bool) => set({ isProcessing: bool }),
  setResult: (letters) => set({ ocrResult: letters, isProcessing: false }),
  setError: (error) => set({ error, isProcessing: false }),
  reset: () =>
    set({
      file: null,
      previewUrl: null,
      isProcessing: false,
      ocrResult: null,
      error: null
    })
}));
