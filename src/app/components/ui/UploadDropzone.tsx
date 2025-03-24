"use client";

import { useRef } from "react";

interface UploadDropzoneProps {
  onFileSelected: (file: File, previewUrl: string) => void;
}

export default function UploadDropzone({
  onFileSelected
}: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      onFileSelected(file, reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div
      onClick={() => inputRef.current?.click()}
      className="border-dashed border-2 border-gray-300 rounded-xl p-6 text-center cursor-pointer hover:bg-gray-50"
    >
      <p className="text-gray-600">
        Click to upload a handwriting image (PNG or JPG)
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
