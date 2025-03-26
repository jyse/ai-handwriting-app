"use client";
import { useRef } from "react";
import { useUploadStore } from "../../state/useUploadStore";

export default function UploadDropzone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const setFile = useUploadStore((s) => s.setFile);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (PNG or JPG).");
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      const previewUrl = reader.result as string;
      setFile(file, previewUrl);

      // Wrap the async logic in an immediately invoked async function
      (async () => {
        const formData = new FormData();
        formData.append("image", file);

        try {
          const res = await fetch("/api/process", {
            method: "POST",
            body: formData
          });

          if (!res.ok) {
            const text = await res.text(); // read the actual HTML error
            console.error("❌ Server responded with error HTML:", text);
            throw new Error("Server returned an error");
          }

          const result = await res.json();
          console.log("🧠 OCR result:", result); // TODO: Store in Zustand
        } catch (err) {
          console.error("Error in OCR fetch:", err);
        }
      })();
    };
    reader.readAsDataURL(file);
  };

  return (
    <div
      onClick={() => inputRef.current?.click()}
      className="border-dashed border-2 border-secondary/50 rounded-xl p-6 text-center cursor-pointer hover:bg-secondary/10 transition"
    >
      <p className="text-tertiary text-sm">
        Click or drag to upload a handwriting image (.png or .jpg)
      </p>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        ref={inputRef}
        className="hidden"
      />
    </div>
  );
}
