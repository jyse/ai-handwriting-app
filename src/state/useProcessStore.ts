import { create } from "zustand";

interface ProcessStore {
  ocrResult: string[] | null;
  missingLetters: string[];
  isAnalyzing: boolean;
  isComplete: boolean;
  shouldUseAICompletion: boolean;
  processingPhase: "idle" | "ocr" | "checking" | "done";

  setOCRResult: (result: string[]) => void;
  setMissingLetters: (missing: string[]) => void;
  setAnalyzing: (isAnalyzing: boolean) => void;
  setComplete: (isComplete: boolean) => void;
  setUseAICompletion: (useAI: boolean) => void;
  setProcessingPhase: (step: "idle" | "ocr" | "checking" | "done") => void;
  reset: () => void;
}

export const useProcessStore = create<ProcessStore>((set) => ({
  ocrResult: null,
  missingLetters: [],
  isAnalyzing: false,
  isComplete: false,
  shouldUseAICompletion: false,
  processingPhase: "idle",

  setOCRResult: (result) => set({ ocrResult: result }),
  setMissingLetters: (missing) => set({ missingLetters: missing }),
  setAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setComplete: (isComplete) => set({ isComplete }),
  setUseAICompletion: (useAI) => set({ shouldUseAICompletion: useAI }),
  setProcessingPhase: (step) => set({ processingPhase: step }),

  reset: () =>
    set({
      ocrResult: null,
      missingLetters: [],
      isAnalyzing: false,
      isComplete: false,
      shouldUseAICompletion: false,
      processingPhase: "idle",
    })
}));