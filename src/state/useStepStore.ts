import { create } from "zustand";
import { persist } from "zustand/middleware";

type Step = 1 | 2 | 3 | 4 | 5;

interface StepStore {
  step: Step;
  setStep: (step: Step) => void;
  nextStep: () => void;
  prevStep: () => void;
}

export const useStepStore = create<StepStore>()(
  persist(
    (set) => ({
      step: 1,
      setStep: (step) => set({ step }),
      nextStep: () =>
        set((state) => ({ step: Math.min(state.step + 1, 5) as Step })),
      prevStep: () =>
        set((state) => ({ step: Math.max(state.step - 1, 1) as Step }))
    }),
    {
      name: "font-step-storage"
    }
  )
);
