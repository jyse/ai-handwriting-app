"use client";
import Image from "next/image";
import { useUploadStore } from "../../../state/useUploadStore";
import { motion } from "framer-motion";
import UploadAnotherButton from "../../ui/UploadAnotherButton";

export default function UploadPreview() {
  const previewUrl = useUploadStore((s) => s.previewUrl);
  const reset = useUploadStore((s) => s.reset);

  if (!previewUrl) return null;

  return (
    <motion.div
      className="flex flex-col gap-4 items-center"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <p className="text-base font-medium text-green-500">
        ✅ Image uploaded successfully!
      </p>

      <div>
        <p className="text-tertiary">🔎 Please ensure:</p>
        <ul className="list-disc pl-6 text-left text-tertiary text-sm">
          <li>👀 Handwriting is clearly visible</li>
          <li>🖋️ Dark ink on a light background</li>
          <li>🌤️ No shadows or folds on paper</li>
        </ul>
      </div>

      <p className="text-sm text-quartiary italic">
        🧠 Check if it is a clean sample! Our AI will analyze this in the next step.
      </p>

      <Image
        src={previewUrl}
        alt="Preview"
        className="w-full max-w-xl h-auto object-contain rounded-xl shadow-md border border-secondary/20"
        width={800}
        height={600}
        unoptimized
        priority
      />

    <UploadAnotherButton/>
    </motion.div>
  );
}
