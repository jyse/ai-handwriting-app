"use client";

import { useStepStore } from "../../state/useStepStore";
import { useUploadStore } from "../../state/useUploadStore";
import clsx from "clsx";

export default function StepNavigation() {
  const { step, nextStep, prevStep } = useStepStore();
  const previewUrl = useUploadStore((state) => state.previewUrl);

  // Disable “Next” only if on Step 1 and no file uploaded
  const isNextDisabled = step === 1 && !previewUrl;

  return (
    <div className="flex justify-between w-full max-w-xl mt-6">
      {/* Hide Back button on first step */}
      {step > 1 ? (
        <button
          onClick={prevStep}
          className="px-4 py-2 border border-secondary rounded text-sm text-secondary hover:bg-secondary hover:text-bg transition"
        >
          ← Back
        </button>
      ) : (
        <div />
      )}

      <button
        onClick={nextStep}
        className={clsx(
          "px-4 py-2 bg-primary text-white rounded text-sm transition",
          isNextDisabled && "opacity-50 cursor-not-allowed"
        )}
        disabled={isNextDisabled}
      >
        Next →
      </button>
    </div>
  );
}
