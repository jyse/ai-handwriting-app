import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("image") as File;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  // For now: return mock data
  return NextResponse.json({
    letters: ["A", "B", "C", "D", "E", "F", "G"]
  });
}
