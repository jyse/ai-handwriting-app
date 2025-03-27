"use client";
import { useEffect } from "react";
import { useUploadStore } from "../../state/useUploadStore";
import { useProcessStore } from "../../state/useProcessStore";
import { useStepStore } from "../../state/useStepStore";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import UploadAnotherButton from "../ui/UploadAnotherButton";

export default function Process() {
  const file = useUploadStore((s) => s.file);
  const nextStep = useStepStore((s) => s.nextStep);

  const {
    setOCRResult,
    setMissingLetters,
    setComplete,
    setProcessingPhase,
    reset: resetProcess,
  } = useProcessStore();

  useEffect(() => {
    if (!file) return;

    const analyze = async () => {
      setProcessingPhase("ocr");

      const formData = new FormData();
      formData.append("image", file);

      try {
        const res = await fetch("/api/process", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();

        if (!data || !data.letters) {
          throw new Error(data?.error || "No letters returned from AI.");
        }

        setOCRResult(data.letters);
        setProcessingPhase("checking")

        setTimeout(() => {


          if (data.isComplete) {
            setComplete(true);
            setProcessingPhase("done");
            toast.success("✅ Font contains all required characters!");
            setTimeout(() => nextStep(), 1200);
          } else {
            setComplete(false);
            setMissingLetters(data.missing || []);
            setProcessingPhase("done");
            toast.error("⚠️ Some characters are missing.");
          }
        }, 800)
      } catch (err: any) {
        console.error("❌ AI error:", err.message);
        resetProcess(); // optional: wipe process state
        setProcessingPhase("idle");
        toast.error("Something went wrong during processing.");
      }
    };

    analyze();
  }, [file]);

  const processingPhase = useProcessStore((s) => s.processingPhase);
  const isComplete = useProcessStore((s) => s.isComplete)
  const missingLetters = useProcessStore((s) => s.missingLetters);

  return (
    <section className="flex flex-col items-center gap-6 text-center max-w-xl mx-auto px-4">
      <h1 className="text-3xl font-heading text-primary">
        Step 2: Processing Handwriting
      </h1>

      {processingPhase === "ocr" && (
        <motion.p className="text-secondary text-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          🧠 Analyzing handwriting sample and extracting characters…
        </motion.p>
      )}

      {processingPhase === "checking" && (
        <motion.p className="text-secondary text-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          🔍 Checking if all required characters are present…
        </motion.p>
      )}

      {processingPhase === "done" && isComplete && (
        <motion.p className="text-green-500 text-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          ✅ All characters are present! Moving to preview...
        </motion.p>

      )}

      {processingPhase === "done" && !isComplete && (
        <motion.div
          className="text-yellow-400 text-sm flex flex-col items-center gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          ⚠️ Your sample is missing some characters. You can upload a new image or let AI generate the rest in your style.
          {missingLetters.length > 0 && (
            <ul className="text-sm text-muted-foreground grid grid-cols-4 gap-2 mt-2">
              {missingLetters.map((char) => (
                <li
                  key={char}
                  className="bg-muted rounded px-2 py-1 text-center border border-dashed"
                >
                  {char === " " ? <span className="italic text-xs text-muted-foreground">[space]</span> : char}
                </li>
              ))}
            </ul>
          )}
          <UploadAnotherButton />
        </motion.div>
      )}
    </section>
  );
}
