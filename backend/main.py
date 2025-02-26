import os
import uuid
import shutil
import subprocess
import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps

app = FastAPI()

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("fonts", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
print("🐲[INFO] Directories created: uploads/, fonts/, templates/, static/")

# Load pre-trained AI model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
print("[SUCCESS] 🤖AI model loaded!")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    print("[INFO] 🌳Root endpoint accessed")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/template", response_class=FileResponse)
async def get_template():
    template_path = "static/handwriting_template.pdf"
    if not os.path.exists(template_path):
        generate_template()
    return FileResponse(template_path)

def generate_template():
    """Generate a PDF template with a grid for writing characters"""
    # This is a placeholder - you would normally use a library like reportlab
    # For now, just create a simple file to indicate functionality
    with open("static/handwriting_template.pdf", "w") as f:
        f.write("Template placeholder - replace with actual PDF generation")
    print("[INFO] Generated character grid template")

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), email: str = Form(None)):
    print(f"[INFO] Received file ✍️📝: {file.filename}")
    print(f"[INFO] Email address: {email if email else 'Not provided'}")
    
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
        # Open and pre-process the image
        image = Image.open(file_path).convert("RGB")
        # Enhance contrast to make handwriting more visible
        image = ImageOps.autocontrast(image, cutoff=1)
        
        # Check if the image is vertical and rotate
        width, height = image.size
        if height > width:
            print("[INFO] 🤸‍♀️🖼️ Rotating image to landscape mode.")
            image = image.rotate(-90, expand=True)
            # Save the rotated image
            image.save(file_path)

        # Optional OCR to validate image has text content
        recognized_text = recognize_handwriting(image)
        if not recognized_text or not recognized_text.strip():
            print("[WARNING] ⚠️ OCR couldn't detect clear text.")
            # We'll proceed anyway since we're using grid-based extraction

        # The most important step: extract individual characters
        success = extract_characters(file_path, user_letters_dir)
        if not success:
            return {"error": "Failed to extract characters from the image"}

    except Exception as e:
        print(f"[ERROR] 👹 Failed to process image: {e}")
        return {"error": "👹 Failed to process handwriting"}

    # Step 2: Fine-Tune Model on User Data (optional, can be removed)
    try:
        if recognized_text and recognized_text.strip():
            fine_tune_model(image, recognized_text)
            print("[SUCCESS] Fine-tuning complete 🏆")
    except Exception as e:
        print(f"[WARNING] Fine-tuning skipped: {e}")
        # Continue anyway since this is optional

    # Step 3: Convert Extracted Characters into a Font
    try:
        # Generate a font using all available SVG files, not just OCR text
        font_path = generate_font(user_letters_dir)
        if isinstance(font_path, dict) and "error" in font_path:
            return font_path  # Return the error
        print(f"[SUCCESS] Font generated: {font_path}")
    except Exception as e:
        print(f"[ERROR] 👹 Font generation failed: {e}")
        return {"error": "👹 Font generation failed"}

    # Return the path to download the font
    return {"font_url": os.path.basename(font_path)}

def recognize_handwriting(image):
    """Use OCR to recognize text (mainly for validation)"""
    print("[INFO] 🌀 Running handwriting recognition...")
    try:
        inputs = processor(image, return_tensors="pt")
        generated_text = model.generate(**inputs)
        recognized_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
        print(f"[INFO] 👀✍️Recognized text: {recognized_text}")
        return recognized_text
    except Exception as e:
        print(f"[WARNING] OCR recognition error: {e}")
        return ""

def extract_characters(image_path, output_dir):
    """Extract individual characters from a grid template"""
    print("[INFO] Extracting individual characters...")
    
    try:
        # Load the image using OpenCV for better grid detection
        img = cv2.imread(image_path)
        if img is None:
            print("[ERROR] Failed to load image")
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Define the expected grid parameters (adjust based on your template)
        # Assuming a 6x9 grid for lowercase and uppercase letters
        grid_rows, grid_cols = 6, 9
        
        # Calculate approximate cell size
        height, width = thresh.shape
        cell_height = height // grid_rows
        cell_width = width // grid_cols
        
        # Define the characters we're extracting (a-z, A-Z)
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        char_index = 0
        successful_chars = []
        
        # Process each cell in the grid
        for row in range(grid_rows):
            for col in range(grid_cols):
                if char_index >= len(chars):
                    break
                    
                # Extract the cell region
                cell_char = chars[char_index]
                y_start = row * cell_height
                y_end = (row + 1) * cell_height
                x_start = col * cell_width
                x_end = (col + 1) * cell_width
                
                cell = thresh[y_start:y_end, x_start:x_end]
                
                # Save as temporary bitmap
                temp_path = os.path.join(output_dir, f"{cell_char}.bmp")
                cv2.imwrite(temp_path, cell)
                
                # Convert to SVG using potrace if installed
                svg_path = os.path.join(output_dir, f"{cell_char}.svg")
                
                try:
                    # Try using potrace for bitmap to SVG conversion
                    subprocess.run(["potrace", temp_path, "-s", "-o", svg_path], check=True)
                    print(f"[INFO] Generated SVG for '{cell_char}'")
                    successful_chars.append(cell_char)
                except FileNotFoundError:
                    # If potrace isn't installed, create a simple SVG placeholder
                    create_simple_svg(cell, svg_path, cell_char)
                    successful_chars.append(cell_char)
                except Exception as e:
                    print(f"[WARNING] Failed to create SVG for '{cell_char}': {e}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                char_index += 1
        
        print(f"[SUCCESS] Successfully extracted {len(successful_chars)} characters: {''.join(successful_chars)}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Character extraction failed: {e}")
        return False

def create_simple_svg(cell_img, svg_path, char):
    """Create a simple SVG if potrace isn't available"""
    # Get the contours of the character
    contours, _ = cv2.findContours(cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a simple SVG with the contours
    height, width = cell_img.shape
    
    with open(svg_path, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n')
        f.write(f'  <title>Character {char}</title>\n')
        
        # Add each contour as a path
        for contour in contours:
            if len(contour) > 2:  # Only use contours with sufficient points
                path = "  <path d=\"M"
                first_point = True
                
                for point in contour:
                    x, y = point[0]
                    if first_point:
                        path += f"{x},{y} "
                        first_point = False
                    else:
                        path += f"L{x},{y} "
                
                path += "Z\" fill=\"black\" />\n"
                f.write(path)
        
        f.write('</svg>')
    
    print(f"[INFO] Created simple SVG for '{char}'")

def fine_tune_model(image, text):
    """Fine-tune the OCR model (optional step)"""
    if not text.strip():
        print("[WARNING] Skipping fine-tuning: No valid text detected.")
        return

    print("[INFO] Fine-tuning model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    inputs = processor(image, return_tensors="pt")
    labels = processor.tokenizer(text, return_tensors="pt").input_ids

    # Ensure required token IDs are set
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id 
            if hasattr(processor.tokenizer, "cls_token_id") 
            else processor.tokenizer.bos_token_id
        )

    if model.config.pad_token_id is None:
        if hasattr(processor.tokenizer, "pad_token_id") and processor.tokenizer.pad_token_id is not None:
            model.config.pad_token_id = processor.tokenizer.pad_token_id
        else:
            print("[WARNING] ⚠️ `pad_token_id` is missing! Fine-tuning may not work correctly.")

    # Compute loss and backpropagation
    try:
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        print(f"[SUCCESS] 🏆 Fine-tuning completed successfully! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"[WARNING] Fine-tuning skipped: {e}")

def generate_font(letters_dir):
    """Generate a font from extracted character SVGs"""
    print("[INFO] Generating font...")

    # Create a unique font name for this generation
    font_name = f"handwriting_font_{uuid.uuid4().hex[:8]}"
    script_path = f"generate_font_{font_name}.pe"
    font_path = os.path.join("fonts", f"{font_name}.ttf")

    # Get all available SVG files
    svg_files = [f for f in os.listdir(letters_dir) if f.endswith('.svg')]
    if not svg_files:
        print("[ERROR] No SVG files found to generate font")
        return {"error": "No character SVGs were successfully created"}

    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")
        f.write(f'SetFontNames("{font_name}", "{font_name}", "{font_name}", "Regular", "Generated Handwriting", "{uuid.uuid4().hex}");\n')
        f.write('SetTTFName(0x409, 1, "Generated Handwriting");\n')
        f.write('SetTTFName(0x409, 2, "Regular");\n')
        
        missing_letters = []
        successful_imports = 0

        # Import available SVGs
        all_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for char in all_chars:
            svg_path = os.path.join(letters_dir, f"{char}.svg")
            if os.path.exists(svg_path):
                f.write(f'Select(Unicode({ord(char)}));\n')
                f.write(f'Import("{svg_path}");\n')
                f.write('RemoveOverlap();\n')
                f.write('Simplify();\n')
                f.write('CorrectDirection();\n')
                successful_imports += 1
                print(f"[INFO] Added letter: {char} from {svg_path}")
            else:
                missing_letters.append(char)
                print(f"[WARNING] Missing SVG for letter: {char}")

        # Set font properties
        f.write('SelectAll();\n')
        f.write('SetWidth(1000);\n')  # Standard width
        f.write('CenterInWidth();\n')  # Center each glyph
        
        # Add font metadata
        f.write('SetFontOrder(2);\n')  # TrueType
        f.write('SetOS2Value("Weight", 400);\n')  # Regular weight
        f.write('SetOS2Value("Width", 5);\n')  # Medium width
        
        # Save the font
        f.write(f'Generate("{font_path}");\n')
        f.write("Quit(0);\n")

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Log missing letters but still generate font
    if missing_letters:
        print(f"[WARNING] 🚨 Missing SVGs for letters: {', '.join(missing_letters)}")

    if successful_imports == 0:
        return {"error": "No character SVGs were successfully imported"}

    # Run the FontForge script
    print("[INFO] Running FontForge script...")
    try:
        subprocess.run(["fontforge", "-script", script_path], check=True)
        print("[SUCCESS] FontForge script executed successfully!")
        # Clean up the script file
        os.remove(script_path)
        return font_path
    except FileNotFoundError:
        print("[ERROR] ❌ FontForge is not installed or not in PATH!")
        return {"error": "FontForge not found. Please install it via Homebrew (mac) or apt-get (Linux)."}
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 👹 FontForge script execution failed: {e}")
        return {"error": f"FontForge execution failed: {e}"}

# Add a route to download the generated font
@app.get("/fonts/{font_name}")
async def get_font(font_name: str):
    font_path = os.path.join("fonts", font_name)
    if os.path.exists(font_path):
        return FileResponse(font_path, media_type="font/ttf", filename=font_name)
    return {"error": "Font not found"}