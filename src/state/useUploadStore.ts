import { create } from "zustand";

interface UploadStore {
  file: File | null;
  previewUrl: string | null;
  ocrResult: string[] | null;
  isProcessing: boolean;
  error: string | null;

  setFile: (file: File, previewUrl: string) => void;
  setResult: (letters: string[]) => void;
  setProcessing: (isProcessing: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useUploadStore = create<UploadStore>((set) => ({
  file: null,
  previewUrl: null,
  ocrResult: null,
  isProcessing: false,
  error: null,

  setFile: (file, previewUrl) => set({ file, previewUrl }),
  setResult: (letters) => set({ ocrResult: letters, isProcessing: false }),
  setProcessing: (isProcessing) => set({ isProcessing }),
  setError: (error) => set({ error, isProcessing: false }),
  reset: () =>
    set({
      file: null,
      previewUrl: null,
      ocrResult: null,
      error: null,
      isProcessing: false
    })
}));
