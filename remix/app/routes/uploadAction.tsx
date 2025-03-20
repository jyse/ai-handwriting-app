import { Form } from "@remix-run/react";

export default function UploadPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-2xl font-semibold">Upload Your Handwriting</h1>
      <Form method="post" encType="multipart/form-data">
        <input type="file" name="handwriting" accept="image/*" required />
        <button
          type="submit"
          className="mt-4 px-4 py-2 bg-teal-500 text-white rounded-lg"
        >
          Upload & Process
        </button>
      </Form>
    </div>
  );
}
