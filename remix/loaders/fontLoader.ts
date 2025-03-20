import { json } from "@remix-run/node";
import { getGeneratedFonts } from "~/services/fontService";

export async function loader() {
  const fonts = await getGeneratedFonts();
  return json(fonts);
}
