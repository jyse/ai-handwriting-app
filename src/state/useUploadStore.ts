import { create } from "zustand";

interface UploadStore {
  file: File | null;
  previewUrl: string | null;
  hasOwnership: boolean;

  setFile: (file: File, previewUrl: string) => void;
  setOwnershipConfirmed: (confirmed: boolean) => void;
  reset: () => void;
}

export const useUploadStore = create<UploadStore>((set) => ({
  file: null,
  previewUrl: null,
  isProcessing: false,
  error: null,
  hasOwnership: false, 
  setFile: (file, previewUrl) => set({ file, previewUrl }),
  setOwnershipConfirmed: (confirmed) => set({ hasOwnership: confirmed }), // ✅ fixed

  reset: () =>
    set({
      file: null,
      previewUrl: null,
      hasOwnership: false
    })
}));
