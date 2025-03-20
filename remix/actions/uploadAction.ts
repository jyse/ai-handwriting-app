import { json } from "@remix-run/node";
import { uploadImage } from "~/services/imageService"; // Backend API call

export async function action({ request }) {
  const formData = await request.formData();
  const file = formData.get("handwriting");

  if (!file) {
    return json({ error: "No file uploaded" }, { status: 400 });
  }

  const response = await uploadImage(file);
  return json({ success: true, url: response.url });
}
