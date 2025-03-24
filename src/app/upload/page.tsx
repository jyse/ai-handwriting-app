// app/upload/page.tsx
"use client";

import { useState } from "react";
import UploadDropzone from "../../components/ui/UploadDropzone";
import ImagePreview from "../../components/ui/ImagePreview";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  return (
    <main className="p-8 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Upload Handwriting Sample</h1>

      <UploadDropzone
        onFileSelected={(file, url) => {
          setSelectedFile(file);
          setPreviewUrl(url);
        }}
      />

      {previewUrl && <ImagePreview src={previewUrl} />}
    </main>
  );
}
