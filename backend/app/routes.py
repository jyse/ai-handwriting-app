import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from app.utils.processing import process_image, recognize_handwriting
from app.utils.file_handling import save_uploaded_file
from app.utils.font_generation import generate_font  
from app.utils.processing import extract_letters
from app.utils.processing import fine_tune_model  
from app.utils.processing import convert_png_to_svg  
from PIL import Image
import traceback

router = APIRouter()
UPLOAD_DIR = "uploads"

@router.post("/upload")
async def upload_image(file: UploadFile = File(...), email: str = Form(None)):
    """Process uploaded handwriting image and convert to font"""
    try:
        # Step 1: Save uploaded file
        session_id, file_path = save_uploaded_file(file, UPLOAD_DIR)
        user_letters_dir = os.path.join(UPLOAD_DIR, session_id, "letters")
        os.makedirs(user_letters_dir, exist_ok=True)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File was not saved correctly")

        # Step 2: Process image and recognize handwriting
        try:
            image = process_image(file_path)
            processed_image_path = os.path.join(UPLOAD_DIR, session_id, "processed_image.png")
            image.save(processed_image_path)
        except Exception as e:
            print(f"⚠️ Image processing error: {e}")
            raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

        # Now pass the processed image file path to extract_letters
        try:
            extracted_letters = extract_letters(processed_image_path, user_letters_dir)
            print(f"🐲 Extracted letters count: {len(extracted_letters)}")
            
            if not extracted_letters: 
                raise HTTPException(status_code=422, detail="No letters were extracted from handwriting image")
        except Exception as e:
            print(f"⚠️ Letter extraction error: {e}")
            raise HTTPException(status_code=422, detail=f"Letter extraction failed: {str(e)}")
        
        # Step 3: Recognize handwriting text
        try:
            recognized_texts = recognize_handwriting(extracted_letters)
            print(f"🔠 Recognized text: {recognized_texts}")
            
            if not recognized_texts:
                raise HTTPException(status_code=422, detail="No text was recognized from the handwriting")
        except Exception as e:
            print(f"⚠️ Text recognition error: {e}")
            raise HTTPException(status_code=422, detail=f"Text recognition failed: {str(e)}")

        # 🛠 Convert extracted letter paths to images for fine-tuning
        letter_images = []
        for letter_file in extracted_letters:
            try:
                letter_images.append(Image.open(letter_file).convert("RGB"))
            except Exception as e:
                print(f"⚠️ Warning: Could not open letter image {letter_file}: {e}")

        # Step 4: Fine-Tune the Model on the User's handwriting (Optional)
        try:
            if letter_images and recognized_texts:
                fine_tune_result = fine_tune_model(letter_images, recognized_texts)
                print(f"🤖 Fine-Tuning Result: {fine_tune_result}")
        except Exception as e:
            print(f"⚠️ Fine-tuning failed: {e}, but continuing with font generation")
            # Don't raise an exception here, continue with font generation

        # Convert extracted PNG letters to SVG before generating font
        try:
            convert_png_to_svg(user_letters_dir, user_letters_dir)
        except Exception as e:
            print(f"⚠️ SVG conversion error: {e}")
            # Continue with PNG files if SVG conversion fails

        # Step 5: Generate font from extracted letters
        try:
            font_path = generate_font(recognized_texts, user_letters_dir)
            print(f"🎨 Generated font path: {font_path}")
            
            if not font_path or not os.path.exists(font_path):
                raise HTTPException(status_code=500, detail="Font generation failed")
                
        except Exception as e:
            print(f"⚠️ Font generation error: {e}")
            traceback.print_exc()  # Print the full traceback
            raise HTTPException(status_code=500, detail=f"Font generation failed: {str(e)}")

        # Successfully processed
        response_data = {
            "success": True,
            "font_url": f"/download-font/{os.path.basename(font_path)}"            
        }

        print("🐲 Backend Response:", response_data)
        return JSONResponse(content=response_data, status_code=200)

    except HTTPException as he:
        # Re-raise HTTP exceptions as they already have the proper format
        raise he
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
        return JSONResponse(
            content={"success": False, "error": str(e)}, 
            status_code=500
        )

@router.get("/download-font/{font_name}")
async def download_font(font_name: str):
    """Download generated font"""
    font_path = os.path.join("fonts", font_name)
    
    if not os.path.exists(font_path):
        raise HTTPException(status_code=404, detail="Font not found")
        
    return FileResponse(
        font_path, 
        filename=font_name, 
        media_type="font/ttf"
    )