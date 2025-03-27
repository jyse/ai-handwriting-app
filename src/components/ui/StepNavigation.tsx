"use client";

import { useStepStore } from "../../state/useStepStore";
import { useUploadStore } from "../../state/useUploadStore";
import clsx from "clsx";

export default function StepNavigation() {
  const { step, nextStep, prevStep } = useStepStore();
  const previewUrl = useUploadStore((state) => state.previewUrl);
  const hasOwnership = useUploadStore((state) => state.hasOwnership);

  // 🔐 Disable Next on Step 1 if missing file OR ownership confirmation
  const isNextDisabled = step === 1 && (!previewUrl || !hasOwnership);

  return (
    <div className="flex justify-between w-full max-w-xl mt-6">
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
          "px-4 py-2 rounded text-sm transition",
          isNextDisabled
            ? "bg-muted text-muted-foreground cursor-not-allowed opacity-50"
            : "bg-primary text-white hover:bg-primary/80"
        )}
        disabled={isNextDisabled}
      >
        Next →
      </button>
    </div>
  );
}
