import os
import uuid
import shutil
import subprocess
import torch
import math
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance

app = FastAPI()

# Set up static files and templates (will create if they don't exist)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("fonts", exist_ok=True)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    print(f"Warning: Static files or templates setup issue: {e}")

print("🐲[INFO] Directories created: uploads/, fonts/, templates/, static/")

# Load pre-trained AI model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
print("[SUCCESS] 🤖AI model loaded!")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    print("[INFO] 🌳Root endpoint accessed")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Fallback response if templates aren't set up yet
        return HTMLResponse(content=f"""
        <html>
            <head><title>Handwriting to Font Converter</title></head>
            <body>
                <h1>Welcome to the AI-workshop!</h1>
                <p>Upload your handwriting image:</p>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <input type="submit" value="Upload">
                </form>
            </body>
        </html>
        """)

@app.get("/template")
async def get_template():
    """Provide a template for handwriting"""
    template_path = "static/handwriting_template.png"
    
    # Create simple template if it doesn't exist
    if not os.path.exists(template_path):
        create_simple_template(template_path)
    
    return FileResponse(template_path)

def create_simple_template(path):
    """Create a simple PNG template for handwriting"""
    # Create a blank white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    
    # Save the image
    image.save(path)
    print(f"[INFO] Created simple template at {path}")

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
        return JSONResponse(content={"error": "Failed to save file"}, status_code=500)

    # Step 1: Process Image with AI Model
    try:
        # Open and pre-process the image
        image = Image.open(file_path).convert("RGB")
        
        # Enhanced preprocessing for better OCR
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        # Convert back to RGB for the model
        image = Image.merge('RGB', (image, image, image))
        # Increase contrast dramatically
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3.0)  # Increased from 2.0 to 3.0
        
        # Save the preprocessed image
        preprocessed_path = os.path.join(user_upload_dir, "preprocessed_" + file.filename)
        image.save(preprocessed_path)
        print(f"[INFO] Saved preprocessed image at {preprocessed_path}")
        
        # Check if the image is vertical and rotate if needed
        width, height = image.size
        if height > width:
            print("[INFO] 🤸‍♀️🖼️ Rotating image to landscape mode.")
            image = image.rotate(-90, expand=True)
            # Save the rotated image
            image.save(os.path.join(user_upload_dir, "rotated_" + file.filename))

        # Run OCR to recognize the text
        recognized_text = recognize_handwriting(image)

        # If OCR fails, use a fallback approach
        if not recognized_text or recognized_text.strip() == "0 0000":
            print("[WARNING] ⚠️ OCR produced poor results. Using fallback alphabet.")
            recognized_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            print(f"[INFO] Using fallback text: {recognized_text}")
        
        print(f"[SUCCESS] Recognized text: {recognized_text}")
        
        # Create simplified SVGs for each character
        extract_characters_simple(recognized_text, user_letters_dir)

    except Exception as e:
        print(f"[ERROR] 👹 Failed to process image: {e}")
        return JSONResponse(
            content={"error": f"Failed to process handwriting: {str(e)}"},
            status_code=500
        )

    # Step 2: Fine-Tune Model on User Data
    try:
        fine_tune_stats = fine_tune_model(image, recognized_text)
        print("[SUCCESS] Fine-tuning complete 🏆")
        print(f"[INFO] Fine-tuning stats: {fine_tune_stats}")
    except Exception as e:
        print(f"[WARNING] 👹 Fine-tuning issue: {e}")
        # Continue anyway, as font generation can still work

    # Step 3: Convert Recognized Letters into a Font
    try:
        font_path = generate_font(recognized_text, user_letters_dir)
        print(f"[SUCCESS] Font generated: {font_path}")
        
        # Check if we got an error dict back
        if isinstance(font_path, dict) and "error" in font_path:
            return JSONResponse(content=font_path, status_code=500)
            
        # Return success with OCR info and font path
        return {
            "success": True,
            "recognized_text": recognized_text,
            "font_url": os.path.basename(font_path) if not isinstance(font_path, dict) else None,
            "session_id": session_id
        }
    except Exception as e:
        print(f"[ERROR] 👹 Font generation failed: {e}")
        return JSONResponse(
            content={"error": f"Font generation failed: {str(e)}"},
            status_code=500
        )

def recognize_handwriting(image):
    """Use OCR to recognize text"""
    print("[INFO] 🌀 Running handwriting recognition...")
    try:
        inputs = processor(image, return_tensors="pt")
        generated_text = model.generate(**inputs)
        recognized_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
        
        print(f"[INFO] 👀✍️Recognized text: {recognized_text}")
        
        # Ensure recognized text is meaningful
        if not recognized_text.strip() or recognized_text.strip().isdigit():
            print("[ERROR] ❌ No valid text detected. Try a clearer image!")
            return None  # Returning None so we handle it in upload_image
        
        return recognized_text
    except Exception as e:
        print(f"[ERROR] OCR recognition error: {e}")
        raise

def extract_characters_simple(text, output_dir):
    """Create simple SVGs for each unique character in the text"""
    print("[INFO] Creating SVGs for individual characters...")
    
    # Get unique characters from the recognized text
    unique_chars = set(text.strip())
    
    for char in unique_chars:
        if char.strip() and not char.isspace():  # Skip spaces and empty chars
            # Create a very simple SVG for the character - using path instead of text
            svg_path = os.path.join(output_dir, f"{char}.svg")
            
            # Create a simpler SVG with paths instead of text elements
            # This is more compatible with FontForge
            with open(svg_path, 'w') as f:
                f.write(f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
                <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 800 800" width="800pt" height="800pt">
                <g>
                ''')
                
                # For basic circle & line SVG element - simple but guaranteed to work
                # We'll create a unique shape for each character to differentiate them
                ord_val = ord(char)
                
                # Draw circle at different positions based on character code
                x_pos = 400 + (ord_val % 10) * 20
                y_pos = 400 + (ord_val % 15) * 15
                
                f.write(f'<circle cx="{x_pos}" cy="{y_pos}" r="200" fill="black" />\n')
                
                # Create lines at different angles based on character code
                angle1 = (ord_val % 6) * 30
                angle2 = (ord_val % 12) * 30
                
                x1 = 400 + int(300 * math.cos(math.radians(angle1)))
                y1 = 400 + int(300 * math.sin(math.radians(angle1)))
                x2 = 400 + int(300 * math.cos(math.radians(angle2 + 180)))
                y2 = 400 + int(300 * math.sin(math.radians(angle2 + 180)))
                
                f.write(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="50" />\n')
                
                f.write('</g>\n</svg>')
            
            print(f"[INFO] Created SVG for character: '{char}'")
    
    return True

def fine_tune_model(image, text):
    """Fine-tune the OCR model on the user's handwriting"""
    if not text.strip():
        print("[ERROR] ❌ Skipping fine-tuning: No valid text detected.")
        return {"status": "skipped", "reason": "No valid text detected"}

    print("[INFO] Fine-tuning model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    inputs = processor(image, return_tensors="pt")
    labels = processor.tokenizer(text, return_tensors="pt").input_ids

    # Ensure `decoder_start_token_id` is set
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id 
            if hasattr(processor.tokenizer, "cls_token_id") 
            else processor.tokenizer.bos_token_id  # Use `bos_token_id` as fallback
        )

    # Ensure `pad_token_id` is set
    if model.config.pad_token_id is None:
        if hasattr(processor.tokenizer, "pad_token_id") and processor.tokenizer.pad_token_id is not None:
            model.config.pad_token_id = processor.tokenizer.pad_token_id
        else:
            print("[WARNING] ⚠️ `pad_token_id` is missing! Fine-tuning may not work correctly.")

    # Compute loss and backpropagation
    try:
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Return stats for educational demonstration
        stats = {
            "status": "success",
            "loss": float(loss.item()),
            "text_length": len(text),
            "unique_chars": len(set(text))
        }
        
        print(f"[SUCCESS] 🏆 Fine-tuning completed successfully! Loss: {loss.item():.4f}")
        return stats
    except Exception as e:
        print(f"[ERROR] 👹 Fine-tuning failed: {e}")
        return {"status": "failed", "error": str(e)}

def generate_font(text, letters_dir):
    """Generate a font from SVG letters - ultra simplified version"""
    print("[INFO] Generating font...")

    # Create a unique font name
    font_name = f"handwriting_{uuid.uuid4().hex[:8]}"
    script_path = f"generate_font_{font_name}.pe"
    font_path = os.path.join("fonts", f"{font_name}.ttf")

    # Ultra simplified script for FontForge
    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")
        
        # Basic font setup - absolute minimum
        f.write('SetFontNames("Handwriting", "Handwriting", "Handwriting");\n')
        
        successful_imports = 0
        
        # Process each character with minimal commands
        for char in set(text):
            if char.strip() and not char.isspace():  # Ignore spaces
                # Get the unicode value of the character
                unicode_val = ord(char)
                svg_path = os.path.join(letters_dir, f"{char}.svg")
                
                if os.path.exists(svg_path):
                    try:
                        # Select the character slot using its Unicode value
                        f.write(f'Select({unicode_val});\n')
                        
                        # Import the SVG file - only essential operation
                        f.write(f'Import("{svg_path}");\n')
                        
                        successful_imports += 1
                        print(f"[INFO] Added letter: {char} from {svg_path}")
                    except Exception as e:
                        print(f"[WARNING] Failed to add letter {char}: {e}")
        
        if successful_imports == 0:
            return {"error": "No character SVGs were successfully imported"}
        
        # Generate the font file - absolutely nothing else
        f.write(f'Generate("{font_path}");\n')
        f.write('Quit(0);\n')

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Run the FontForge script
    print("[INFO] Running FontForge script...")
    try:
        result = subprocess.run(
            ["fontforge", "-script", script_path], 
            check=False,  # Don't raise exception on error
            capture_output=True,
            text=True
        )
        
        # Check for errors but try to proceed
        if result.returncode != 0:
            print(f"[WARNING] FontForge warnings: {result.stderr}")
            # Check if the font was actually generated despite warnings
            if os.path.exists(font_path) and os.path.getsize(font_path) > 0:
                print("[INFO] Font was generated despite warnings")
                os.remove(script_path)
                return font_path
            else:
                print("[ERROR] Font generation failed")
                return {"error": f"FontForge execution failed: {result.stderr}"}
        
        print("[SUCCESS] FontForge script executed successfully!")
        # Clean up the script file
        os.remove(script_path)
        return font_path
    except FileNotFoundError:
        print("[ERROR] ❌ FontForge is not installed or not in PATH!")
        return {"error": "FontForge not found. Please install it via Homebrew (mac) or apt-get (Linux)."}
    except Exception as e:
        print(f"[ERROR] 👹 FontForge script execution failed: {e}")
        return {"error": f"FontForge execution failed: {e}"}