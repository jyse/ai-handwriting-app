"use client";
import { useEffect, useState } from "react";
import { useUploadStore } from "../../state/useUploadStore";
import { useStepStore } from "../../state/useStepStore";
import { motion } from "framer-motion";

export default function Process() {
  const file = useUploadStore((s) => s.file);
  const nextStep = useStepStore((s) => s.nextStep);

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    const processImage = async () => {
      if (!file || loading) return;

      setLoading(true);
      setSuccess(false);

      const formData = new FormData();
      formData.append("image", file);

      try {
        const res = await fetch("/api/process", {
          method: "POST",
          body: formData
        });

        const data = await res.json();
        console.log("✅ AI Output:", data);

        setSuccess(true);

        // Optional: Save data.letters to Zustand for later use.
        setTimeout(() => {
          nextStep(); // Auto go to Preview
        }, 1200);
      } catch (err) {
        console.error("❌ Error processing image:", err);
        alert("Something went wrong while processing your handwriting.");
      } finally {
        setLoading(false);
      }
    };

    processImage();
  }, [file]);

  return (
    <section className="flex flex-col items-center gap-6 text-center max-w-xl mx-auto px-4">
      <h1 className="text-3xl font-heading text-primary">
        Step 2: Processing Handwriting
      </h1>

      <p className="text-secondary max-w-md">
        We’re using AI to analyze your handwriting and extract the letters. This
        only takes a moment...
      </p>

      {loading && (
        <motion.p
          className="text-quartiary text-sm italic"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          🔍 Analyzing your handwriting...
        </motion.p>
      )}

      {success && (
        <motion.p
          className="text-green-500 text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          ✅ Handwriting processed successfully! Moving to next step...
        </motion.p>
      )}
    </section>
  );
}
