"use client";

import { useStepStore } from "../../state/useStepStore";

export default function StepNavigation() {
  const { step, nextStep, prevStep } = useStepStore();

  return (
    <div className="flex justify-between items-center w-full max-w-md mx-auto px-4">
      {step > 1 ? (
        <button
          onClick={prevStep}
          className="px-4 py-2 rounded bg-secondary text-bg font-medium hover:opacity-90 transition"
        >
          ← Back
        </button>
      ) : (
        <div />
      )}

      {step < 5 ? (
        <button
          onClick={nextStep}
          className="px-4 py-2 rounded bg-primary text-bg font-medium hover:opacity-90 transition"
        >
          Next →
        </button>
      ) : (
        <div />
      )}
    </div>
  );
}
