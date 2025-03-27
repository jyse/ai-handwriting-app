"use client";
import { useUploadStore } from "../../../state/useUploadStore";
import UploadIntro from "./UploadIntro";
import UploadDropzone from "../../ui/UploadDropzone";
import UploadPreview from "./UploadPreview";
import OwnershipConfirm from "./OwnershipConfirm";

export default function Upload() {
  const previewUrl = useUploadStore((s) => s.previewUrl);

  return (
    <section className="flex flex-col gap-6 items-center text-center max-w-xl mx-auto px-4">
      <h1 className="text-3xl font-heading text-primary">
        Step 1: Upload Handwriting
      </h1>

      {!previewUrl ? (
        <>
          <UploadIntro />
          <UploadDropzone />;
        </>
      ) : (
        <>
          <UploadPreview />
          <OwnershipConfirm />
        </>
      )}
    </section>
  );
}
