"use client";
import { useUploadStore } from "../../../state/useUploadStore";

export default function OwnershipConfirm() {
  const hasOwnership = useUploadStore((s) => s.hasOwnership)
  const setOwnershipConfirmed = useUploadStore((s) => s.setOwnershipConfirmed);

  return (
    <label className="flex items-start gap-3 text-left text-sm text-tertiary max-w-md leading-snug cursor-pointer">
      <input
        id="ownership-confirm"
        type="checkbox"
        checked={hasOwnership}
        onChange={(e) => setOwnershipConfirmed(e.target.checked)}
        className="mt-1 accent-primary"
      />
      <span>
        I confirm that I own all of the rights to this handwriting sample and take full
        responsibility for its use and distribution. With giving consent, I am aware that this font will embed this
        declaration in its metadata.
      </span>
    </label>
  );
}
