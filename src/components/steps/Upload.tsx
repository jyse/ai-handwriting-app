"use client";
import Image from "next/image";
import UploadDropzone from "../ui/UploadDropzone";
import { useUploadStore } from "../../state/useUploadStore";
import { motion } from "framer-motion";

export default function Upload() {
  const previewUrl = useUploadStore((s) => s.previewUrl);

  return (
    <section className="flex flex-col gap-6 items-center text-center max-w-xl mx-auto px-4">
      <h1 className="text-3xl font-heading text-primary">
        Step 1: Upload Handwriting
      </h1>
      <p className="text-secondary">
        Start by uploading a handwriting sample. We will convert it into a font.
      </p>

      <UploadDropzone />

      {previewUrl && (
        <motion.div
          className="mt-4 flex flex-col gap-2 items-center"
          initial={{ opacity: 1, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <p className="text-sm">✅ Image uploaded successfully!</p>
          <p className="text-xs text-tertiary mt-2">🔎 Please ensure:</p>
          <ul className="list-disc pl-6 text-left text-xs text-tertiary space-y-1">
            <li>👀 Handwriting is clearly visible</li>
            <li>🖋️ Dark ink on a light background</li>
            <li>🌤️ No shadows or folds on paper</li>
          </ul>

          <Image
            src={previewUrl}
            alt="Preview"
            className="max-w-md max-h-[300px] w-auto rounded-xl shadow"
            width={300}
            height={300}
            unoptimized
            priority
          />
        </motion.div>
      )}
    </section>
  );
}
