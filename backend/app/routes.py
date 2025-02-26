import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from app.utils.processing import process_image, recognize_handwriting
from app.utils.file_handling import save_uploaded_file  # ✅ Import file handling
from app.utils.font_generation import generate_font  # ✅ Import font generation
from app.utils.processing import extract_letters
from app.utils.processing import fine_tune_model  # ✅ Import fine-tuning

router = APIRouter()
UPLOAD_DIR = "uploads"

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Process uploaded handwriting image and convert to font"""
    try:
        # Step 1: Save uploaded file
        session_id, file_path = save_uploaded_file(file, UPLOAD_DIR)
        user_letters_dir = os.path.join(UPLOAD_DIR, session_id, "letters")

        # Step 2: Process image and recognize handwriting
        image = process_image(file_path)
        extracted_letters = extract_letters(file_path, user_letters_dir) 

        if not extracted_letters: 
            raise Exception("No letters were extracted from handwriting image")
        
        # Step 3: Recognize handwriting text
        recognized_text = recognize_handwriting(image)
        print(f"🔠 Recognized text: {recognized_text}")  # ✅ Debugging

        # Step 4: Fine-Tune the Model on the User's handwriting
        fine_tune_result = fine_tune_model(image, recognized_text)
        print(f"🤖 Fine-Tuning Result: {fine_tune_result}")

        # Step 5: Generate font from extracted letters
        font_path = generate_font(recognized_text, user_letters_dir)
        print(f"🎨 Generated font path: {font_path}")  # ✅ Debugging

        # If font generation fails, return an error
        if not font_path or isinstance(font_path, dict):  # Handle errors in font generation
            raise Exception("Font generation failed")

        response_data = {
            "success": True,
            "font_url": f"/download-font/{os.path.basename(font_path)}"            
        }

        print("🐲 Backend Response:", response_data)  # ✅ Log the response before sending it
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        print(f"❌ Error: {e}")  # ✅ Log error
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/download-font/{font_name}")
async def download_font(font_name: str):
    """Download generated font"""
    font_path = os.path.join("fonts", font_name)
    return FileResponse(font_path, filename=font_name, media_type="font/ttf")
