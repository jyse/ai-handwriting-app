"use client";
import Image from "next/image";
import UploadDropzone from "../ui/UploadDropzone";
import { useUploadStore } from "../../state/useUploadStore";
import { motion } from "framer-motion";

export default function Upload() {
  const previewUrl = useUploadStore((s) => s.previewUrl);
  const reset = useUploadStore((s) => s.reset);

  return (
    <section className="flex flex-col gap-6 items-center text-center max-w-xl mx-auto px-4">
      <h1 className="text-3xl font-heading text-primary">
        Step 1: Upload Handwriting
      </h1>

      {!previewUrl ? (
        <>
          <p className="text-base font-medium text-secondary">
            Start by uploading a handwriting sample. We will convert it into a
            font.
          </p>

          <UploadDropzone />
        </>
      ) : (
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
            <p className="text-tertiary ">🔎 Please ensure:</p>
            <ul className="list-disc pl-6 text-left text-tertiary">
              <li>👀 Handwriting is clearly visible</li>
              <li>🖋️ Dark ink on a light background</li>
              <li>🌤️ No shadows or folds on paper</li>
            </ul>
          </div>
          <p className="text-sm text-quartiary italic">
            🧠 Check if it is a clean sample! Our AI will analyze this in the
            next step.
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

          {/* 🔁 Upload Another Button */}
          <button
            onClick={reset}
            className="mt-4 px-3 py-1 border text-sm rounded-md text-primary border-primary hover:border-secondary hover:text-secondary transition-colors duration-200"
          >
            Upload a different image
          </button>
        </motion.div>
      )}
    </section>
  );
}
