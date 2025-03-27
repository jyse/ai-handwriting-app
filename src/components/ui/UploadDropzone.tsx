"use client";
import { useRef } from "react";
import { useUploadStore } from "../../state/useUploadStore";

export default function UploadDropzone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const setFile = useUploadStore((s) => s.setFile);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (PNG or JPG).");
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      const previewUrl = reader.result as string;
      setFile(file, previewUrl); // ✅ Store in Zustand, nothing else
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
