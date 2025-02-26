"use client";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fontDownload, setFontDownload] = useState(null);
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    // Validate file type (only allow images)
    if (!selectedFile.type.startsWith("image/")) {
      alert("Please upload an image file (PNG, JPG, JPEG).");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("email", email);

    try {
      const response = await axios.post(
        "http://localhost:8000/upload",
        formData
      );
      setFontDownload(response.data.font_url);
    } catch (error) {
      alert("Something went wrong. Please try again.");
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
    <div className="flex flex-col items-center p-6 w-full max-w-md mx-auto">
      <h1 className="text-2xl font-bold text-center">
        Upload Your Handwriting
      </h1>

      <input type="file" onChange={handleFileChange} className="mt-4" />

      {preview && (
        <img src={preview} alt="Preview" className="mt-4 w-60 rounded shadow" />
      )}

      <input
        type="email"
        placeholder="Enter your email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="mt-4 p-2 border rounded w-full"
      />

      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded w-full"
        disabled={loading}
      >
        {loading ? "Processing..." : "Upload & Generate Font"}
      </button>

      {fontDownload && (
        <a
          href={fontDownload}
          download="handwriting_font.ttf"
          className="mt-4 text-blue-600 underline"
        >
          Download Your Font
        </a>
      )}

      <button
        onClick={handleShare}
        className="mt-4 px-4 py-2 bg-gray-500 text-white rounded w-full"
      >
        Share on Twitter
      </button>

      <p className="text-sm text-gray-600 text-center mt-4">
        I just turned my handwriting into a font with AI! Try it out here!
      </p>
    </div>
  );
}
