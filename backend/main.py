import os
import uuid
import shutil
import subprocess
import torch
from fastapi import FastAPI, File, UploadFile
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

app = FastAPI()

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("fonts", exist_ok=True)
print("🐲[INFO] Directories created: uploads/ and fonts/")

# Load pre-trained AI model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
print("[SUCCESS] 🤖AI model loaded!")

@app.get("/")
def read_root():
    print("[INFO] 🌳Root endpoint accessed")
    return {"message": "Welcome to the AI-workshop!"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    print(f"[INFO] Received file ✍️📝: {file.filename}")
    
    # Create a unique ID for this user session
    session_id = str(uuid.uuid4())
    print(f"[INFO] 🧘 Generated session ID: {session_id}")
    
    # Create unique directories for this user's processing
    user_upload_dir = f"uploads/{session_id}"
    user_letters_dir = f"uploads/{session_id}/letters"
    os.makedirs(user_upload_dir, exist_ok=True)
    os.makedirs(user_letters_dir, exist_ok=True)
    print(f"[INFO] Created user directories: {user_upload_dir}, {user_letters_dir}")

    # Save the uploaded file
    file_path = os.path.join(user_upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[SUCCESS] File saved at {file_path}")
    except Exception as e:
        print(f"[ERROR] 👹 Failed to save file: {e}")
        return {"error": "👹 Failed to save file"}

    # Step 1: Process Image with AI Model
    try:
        image = Image.open(file_path).convert("RGB")
        
        # Check if the image is vertical and rotate
        width, height = image.size
        if height > width:
          print("[INFO] 🤸‍♀️🖼️ Rotating image to landscape mode for better OCR recognition.")
          image = image.rotate(-90, expand=True)

        recognized_text = recognize_handwriting(image)
        print(f"[SUCCESS] Recognized text: {recognized_text}")
    except Exception as e:
        print(f"[ERROR] 👹 Failed to process image: {e}")
        return {"error": "👹 Failed to process handwriting"}

    # Step 2: Fine-Tune Model on User Data
    try:
        fine_tune_model(image, recognized_text)
        print("[SUCCESS] Fine-tuning complete 🏆")
    except Exception as e:
        print(f"[ERROR] 👹 Fine-tuning failed: {e}")
        return {"error": "👹 Fine-tuning failed"}

    # Step 3: Convert Recognized Letters into a Font
    try:
        font_path = generate_font(recognized_text, user_letters_dir)
        print(f"[SUCCESS] Font generated: {font_path}")
    except Exception as e:
        print(f"[ERROR] 👹 Font generation failed: {e}")
        return {"error": "👹 Font generation failed"}

    return {"font_url": font_path}

def recognize_handwriting(image):
    print("[INFO] 🌀 Running handwriting recognition...")
    inputs = processor(image, return_tensors="pt")
    generated_text = model.generate(**inputs)
    recognized_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
    
    print(f"[INFO] 👀✍️Recognized text: {recognized_text}")
    
    # Ensure recognized text is meaningful
    if not recognized_text.strip() or recognized_text.strip().isdigit():
        print("[ERROR] ❌ No valid text detected. Try a clearer image!")
        return None  # Returning None so we handle it in upload_image

    return recognized_text

def fine_tune_model(image, text):
    # LoRA (Lightweight Fine-Tuning)

    if not text or not text.strip():
        print("[ERROR] ❌ Skipping fine-tuning: No valid text detected.")
        return

    print("[INFO] Fine-tuning model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    inputs = processor(image, return_tensors="pt")
    labels = processor.tokenizer(text, return_tensors="pt").input_ids

    # ✅ Ensure decoder start token is set
    if model.config.decoder_start_token_id is None:
        if hasattr(processor.tokenizer, "cls_token_id"):
            model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        else:
            model.config.decoder_start_token_id = processor.tokenizer.bos_token_id  # Fallback option

    loss = model(**inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    print("[INFO] Fine-tuning completed")

def generate_font(text, letters_dir):
    print("[INFO] Generating font...")

    script_path = "generate_font.pe"
    font_path = os.path.join("fonts", "handwriting_font.ttf")

    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")

        missing_letters = []

        # Import outlines for each letter
        for letter in text:
            svg_path = os.path.join(letters_dir, f"{letter}.svg")
            if os.path.exists(svg_path):
                f.write(f'Select(Unicode({ord(letter)}));\n')
                f.write(f'Import("{svg_path}");\n')
                print(f"[INFO] Added letter: {letter} from {svg_path}")
            else:
                print(f"[WARNING] Missing SVG for letter: {letter}")
                missing_letters.append(letter)

        # Log missing letters but still generate font
        if missing_letters:
            print(f"[WARNING] 👹🚨 Missing SVGs for letters: {', '.join(missing_letters)}")

        # Save the font
        f.write(f'Generate("{font_path}");\n')
        f.write("Quit(0);\n")

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Run the FontForge script
    print("[INFO] Running FontForge script...")
    try:
        subprocess.run(["fontforge", "-script", script_path], check=True)
        print("[SUCCESS] FontForge script executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 👹 FontForge script execution failed: {e}")
        raise

    return font_path
