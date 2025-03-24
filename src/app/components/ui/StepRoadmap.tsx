type StepRoadmapProps = {
  currentStep: number;
};

export default function StepRoadmap({ currentStep }: StepRoadmapProps) {
  const steps = [
    "Upload Image",
    "Process Handwriting",
    "Preview Font",
    "Customize",
    "Download"
  ];

  return (
    <div className="flex items-center justify-center gap-4 py-6">
      {steps.map((label, index) => {
        const step = index + 1;
        const isActive = currentStep === step;
        return (
          <div
            key={label}
            className={`flex flex-col items-center text-sm ${
              isActive ? "text-primary font-semibold" : "text-secondary"
            }`}
          >
            <div
              className={`w-6 h-6 flex items-center justify-center rounded-full border ${
                isActive
                  ? "bg-primary text-white"
                  : "border-secondary text-secondary"
              }`}
            >
              {step}
            </div>
            <span className="mt-1">{label}</span>
          </div>
        );
      })}
    </div>
  );
}
