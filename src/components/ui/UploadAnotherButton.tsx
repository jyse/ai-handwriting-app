"use client";
import { useUploadStore } from "../../state/useUploadStore";

export default function UploadAnotherButton() {
  const reset = useUploadStore((s) => s.reset);

  return (
    <button
      onClick={reset}
      className="mt-4 px-3 py-1 border text-sm rounded-md text-primary border-primary hover:border-secondary hover:text-secondary transition-colors duration-200"
    >
      Upload a different image
    </button>
  );
}
