import { NextRequest, NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("image") as File;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  try {
    const buffer = Buffer.from(await file.arrayBuffer());
    const filePath = path.join(process.cwd(), "tmp", file.name);
    await writeFile(filePath, buffer);
  } catch (err) {
    return NextResponse.json({ error: "Failed to read file" }, { status: 500 });
  }

  // TODO: Replace this mock with real TrOCR/OpenCV logic
  const result = {
    status: "ok",
    letters: ["A", "B", "C"]
  };

  return NextResponse.json(result);
}
