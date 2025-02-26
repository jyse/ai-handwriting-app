"use client";
import { useState } from "react";
import axios from "axios";
import Image from "next/image";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fontDownload, setFontDownload] = useState(null);
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    // Validate file type (only allow images)
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPG, JPEG).");
      return;
    }

    setError(null);
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    if (!email.trim()) {
      setError("Please enter a valid email address.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("email", email);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      console.log("🐲 FULL RESPONSE:", data);

      if (!data || typeof data !== "object") {
        throw new Error("Invalid response from backend");
      }

      if (!data.success) {
        throw new Error(data.error || "Unknown error occurred");
      }

      setFontDownload(data.font_url);
    } catch (error) {
      console.error("👹 ERROR DETAILS:", error);
      setError(error.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleShare = () => {
    const shareText = encodeURIComponent(
      `I just turned my handwriting into a font with AI! Try it out: yourwebsite.com`
    );
    window.open(`https://twitter.com/intent/tweet?text=${shareText}`, "_blank");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6 bg-gray-100">
      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md">
        <h1 className="text-2xl font-bold text-center text-gray-800">
          Handwriting to Font Converter
        </h1>

        <p className="text-sm text-gray-500 text-center mt-2">
          Upload your handwriting and turn it into a custom font!
        </p>

        <div className="mt-4 w-full">
          <input
            type="file"
            onChange={handleFileChange}
            className="w-full p-2 border border-gray-300 rounded-lg"
          />
        </div>

        {preview && (
          <div className="mt-4">
            <Image
              src={preview}
              alt="Preview"
              className="w-full rounded-lg shadow-md"
              width={500}
              height={500}
              unoptimized
            />
          </div>
        )}

        <div className="mt-4 w-full">
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-lg"
          />
        </div>

        {error && (
          <p className="text-sm text-red-500 text-center mt-2">{error}</p>
        )}

        <button
          onClick={handleUpload}
          className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Processing..." : "Upload & Generate Font"}
        </button>

        {fontDownload && (
          <a
            href={fontDownload}
            download="handwriting_font.ttf"
            className="mt-4 text-blue-600 underline text-center block"
          >
            Download Your Font
          </a>
        )}

        <button
          onClick={handleShare}
          className="mt-4 w-full px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition"
        >
          Share on Twitter
        </button>

        <p className="text-sm text-gray-600 text-center mt-4">
          I just turned my handwriting into a font with AI! Try it out here!
        </p>
      </div>
    </div>
  );
}
