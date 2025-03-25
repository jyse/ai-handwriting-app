"use client";
import { useUploadStore } from "../../state/useUploadStore";

export default function Process() {
  const file = useUploadStore((s) => s.file);

  if (!file) {
    return (
      <div className="text-center text-sm text-red-400">
        ⚠️ No image found. Please upload your handwriting first.
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-6 text-center max-w-xl mx-auto px-4">
      <h2 className="text-2xl font-heading text-primary">
        Step 2: Process Handwriting
      </h2>
      <p className="text-secondary">
        Our AI will now analyze your handwriting sample and extract each
        character.
      </p>

      <p className="text-tertiary text-sm">
        🧠 This may take a few seconds depending on image size...
      </p>

      <button
        className="mt-4 px-4 py-2 bg-primary text-white rounded-md shadow hover:bg-primary/90 transition"
        onClick={() => {
          // 👇 We'll hook this up to OCR next
          console.log("Start OCR process...");
        }}
      >
        Start Analysis
      </button>
    </div>
  );
}
