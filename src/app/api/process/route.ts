import { NextRequest, NextResponse } from "next/server";

const EXPECTED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split("");

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("image") as File;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  // 🧠 TODO: Replace this mock with real OCR logic later
  const mockExtractedChars = ["a", "b", "c", "d", "e", "f", "g", "A", "B", "C", "D", "E", "F", "G"];
//   const { extractedText } = await callTrOCR(file);
// const extractedLetters = Array.from(new Set(extractedText.replace(/\s/g, '').split('')));


  const missing = EXPECTED_CHARS.filter((char) => !mockExtractedChars.includes(char));

  return NextResponse.json({
    letters: mockExtractedChars,
    isComplete: missing.length === 0,
    missing,
  });
}
