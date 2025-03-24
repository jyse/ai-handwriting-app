interface Step {
  number: number;
  label: string;
}

const steps: Step[] = [
  { number: 1, label: "Upload Image" },
  { number: 2, label: "Process Handwriting" },
  { number: 3, label: "Preview Font" },
  { number: 4, label: "Customize" },
  { number: 5, label: "Download" }
];

export default function StepRoadmap({ current = 1 }: { current?: number }) {
  return (
    <div className="flex justify-center mt-6 space-x-6">
      {steps.map((step) => {
        const isActive = step.number === current;
        const isComplete = step.number < current;

        return (
          <div
            key={step.number}
            className="flex flex-col items-center text-center"
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold
              ${
                isActive
                  ? "bg-blue-600 text-white"
                  : "bg-zinc-300 dark:bg-zinc-700 text-zinc-800 dark:text-zinc-200"
              }
              transition-all`}
            >
              {step.number}
            </div>
            <span
              className={`text-sm mt-1 ${
                isActive
                  ? "text-blue-600 font-medium"
                  : "text-zinc-500 dark:text-zinc-400"
              }`}
            >
              {step.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
