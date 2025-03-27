"use client";
import { useEffect, useState } from "react";
import { useUploadStore } from "../../state/useUploadStore";
import { useStepStore } from "../../state/useStepStore";
import { motion } from "framer-motion";

export default function Process() {
  const file = useUploadStore((s) => s.file);
  const setResult = useUploadStore((s) => s.setResult);
  const setError = useUploadStore((s) => s.setError);
  const nextStep = useStepStore((s) => s.nextStep);

  const [status, setStatus] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");

  useEffect(() => {
    if (!file) return;

    const processImage = async () => {
      setStatus("loading");

      const formData = new FormData();
      formData.append("image", file);

      try {
        const res = await fetch("/api/process", {
          method: "POST",
          body: formData
        });

        const data = await res.json();

        if (data?.letters) {
          setResult(data.letters);
          setStatus("success");
          setTimeout(() => {
            nextStep();
          }, 1200);
        } else {
          throw new Error(data?.error || "No letters returned");
        }
      } catch (err: any) {
        console.error("❌ OCR error:", err.message);
        setError(err.message || "Unknown error");
        setStatus("error");
      }
    };

    processImage();
  }, [file]);

  return (
    <section className="flex flex-col items-center gap-6 text-center max-w-xl mx-auto px-4">
      <h1 className="text-3xl font-heading text-primary">
        Step 2: Processing Handwriting
      </h1>

      {status === "loading" && (
        <motion.p
          className="text-secondary text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          🧠 Analyzing handwriting…
          <br />
          ✍️ Extracting letters…
          <br />
          🔍 Checking for missing characters…
        </motion.p>
      )}

      {status === "success" && (
        <motion.p
          className="text-green-500 text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          ✅ Handwriting processed successfully! Moving to next step...
        </motion.p>
      )}

      {status === "error" && (
        <p className="text-quartiary text-sm mt-4">
          ❌ Something went wrong. Please go back and try a different image.
        </p>
      )}
    </section>
  );
}
