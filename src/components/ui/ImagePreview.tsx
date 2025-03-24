import Image from "next/image";

interface ImagePreviewProps {
  src: string;
}

export default function ImagePreview({ src }: ImagePreviewProps) {
  return (
    <div className="mt-6">
      <h2 className="text-lg font-semibold mb-2">Preview</h2>
      <Image
        src={src}
        alt="Handwriting Preview"
        className="rounded-lg shadow-md max-w-full"
      />
    </div>
  );
}
