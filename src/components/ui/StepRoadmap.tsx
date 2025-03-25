"use client";
import { useStepStore } from "../../state/useStepStore";
import clsx from "clsx";

const steps = [
  { number: 1, label: "Upload" },
  { number: 2, label: "Process" },
  { number: 3, label: "Preview" },
  { number: 4, label: "Customize" },
  { number: 5, label: "Download" }
];

export default function StepRoadmap() {
  const { step } = useStepStore();
  return (
    <div className="flex justify-center mb-6">
      {steps.map((s, index) => (
        <div key={s.number} className="flex items-center">
          <div
            className={clsx(
              "w-8 h-8 rounded-full border flex items-center justify-center text-sm font-bold",
              step === s.number
                ? "bg-primary text-bg border-primary"
                : "bg-transparent border-secondary text-secondary"
            )}
          >
            {s.number}
          </div>

          {/* Label below circle */}
          <div className="absolute mt-14 text-xs text-center w-24 -ml-8 text-tertiary font-body">
            {s.label}
          </div>

          {/* Connector line */}
          {index !== steps.length - 1 && (
            <div
              className={clsx(
                "w-14 h-[2px] rounded",
                step > s.number ? "bg-primary" : "bg-secondary"
              )}
            />
          )}
        </div>
      ))}
    </div>
  );
}
