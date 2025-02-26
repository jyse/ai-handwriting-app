import os
import uuid
import shutil
import subprocess
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance

app = FastAPI()

# Set up static files and templates (will create if they don't exist)
for directory in ["static", "templates", "uploads", "fonts"]:
    os.makedirs(directory, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")

print("🐲[INFO] Directories created: uploads/, fonts/, templates/, static/")

# Load pre-trained AI model
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("[SUCCESS] 🤖AI model loaded!")
    model_loaded = True
except Exception as e:
    print(f"[WARNING] ⚠️ Could not load AI model: {e}")
    print("[INFO] Will use fallback alphabet instead of OCR")
    model_loaded = False

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
        # Process the image
        processed_image = process_image(file_path)

        # Recognize text if model is loaded
        if model_loaded: 
            recognized_text = recognize_handwriting(processed_image)
            print(f"[SUCCESS] Recognized text: {recognized_text}")
        else:
            # Use fallback if model not loaded
            recognized_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            print(f"[INFO] Using fallback text: {recognized_text}")

        # If OCR text is too short, use a fallback alphabet    
        if not recognized_text or len(recognized_text) < 10:
            print("[WARNING] ⚠️ OCR produced poor results. Using fallback alphabet.")
            recognized_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            print(f"[INFO] Using fallback text: {recognized_text}")

        # Extract handwritten letters
        letter_images = extract_letters(file_path, user_letters_dir)
        if not letter_images: 
            raise ValueError("No letters were extracted!")
        
        # Attempt to convert PNG to SVG (but continue even if it fails)
        try:
            convert_png_to_svg(user_letters_dir, user_letters_dir)
        except Exception as e:
            print(f"[WARNING] SVG conversion issue: {e}")
            print("[INFO] Will continue with PNG files only")
        
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return JSONResponse(content={"error": f"Image processing failed: {str(e)}"}, status_code=500)
    
    # Step 2: Fine-Tune Model (optional)
    if model_loaded:
        try:
            fine_tune_stats = fine_tune_model(processed_image, recognized_text)
            print("[SUCCESS] Fine-tuning complete 🏆")
            print(f"[INFO] Fine-tuning stats: {fine_tune_stats}")
        except Exception as e:
            print(f"[WARNING] 👹 Fine-tuning issue: {e}")
            # Continue anyway, as font generation can still work

    # Step 3: Generate Font
    try:
        font_result = generate_font(recognized_text, user_letters_dir)

        # Check font generation result
        if isinstance(font_result, str) and os.path.exists(font_result):
            print(f"[SUCCESS] Font generated: {font_result}")
            font_url = os.path.join("fonts", os.path.basename(font_result))
            success = True
        else:
            print(f"[WARNING] Font generation produced alternative result: {font_result}")
            font_url = None
            success = False

        # Return result
        return {
            "success": success,
            "recognized_text": recognized_text,
            "font_url": font_url,
            "session_id": session_id,
            "letters_dir": user_letters_dir
        }
    except Exception as e:
        print(f"[ERROR] 👹 Font generation failed: {e}")
        return JSONResponse(
            content={
                "error": f"Font generation failed: {str(e)}",
                "session_id": session_id,
                "letters_dir": user_letters_dir
            },
            status_code=500
        )
    
def process_image(image_path): 
    """Preprocess the image for OCR"""
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3.0)
    return image.convert("RGB")  # Ensure 3-channel image for OCR

def recognize_handwriting(image):
    """Use OCR to recognize text"""
    print("[INFO] 🌀 Running handwriting recognition...")

    if not model_loaded:
        return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    try:
        inputs = processor(image, return_tensors="pt")
        generated_text = model.generate(**inputs)
        return processor.batch_decode(generated_text, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"[WARNING] OCR failed: {e}")
        return ""

def extract_letters(image_path, output_dir):
    """Extract individual handwritten letters from a grid template."""    
    print("[INFO] ✂️ Extracting letters from handwriting grid...")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply adaptive thresholding to better handle grid lines vs. text
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    # Debug info
    print(f"[INFO] Image shape: {image.shape}, unique pixel values: {np.unique(binary)}")
    
    # Try to detect grid cells - we'll look for rectangles
    # First we need to remove some noise with morphological operations
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours - these should be our grid cells and letters
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Found {len(contours)} potential contours")

    # Filter and sort contours by position (top to bottom, left to right)
    valid_contours = []
    min_contour_area = 100  # Adjust based on your image resolution
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Skip very small contours (noise)
        if area < min_contour_area:
            continue
            
        valid_contours.append((x, y, w, h, contour))
    
    # Sort contours by rows, then by columns
    # This assumes the grid cells are arranged in rows
    row_sorted = sorted(valid_contours, key=lambda c: c[1])

    # Make sure output directory exists 
    os.makedirs(output_dir, exist_ok=True)

    letter_files = []

    # Process each contour as a potential cell with a letter
    for i, (x, y, w, h, contour) in enumerate(row_sorted):
        # Extract region containing the letter
        cell = binary[y:y+h, x:x+w]
        
        # Skip empty cells (no significant black pixels)
        if np.sum(cell) < 500:  # Threshold for "significant" content
            print(f"[INFO] Skipping empty cell at ({x}, {y})")
            continue
            
        # Save as PNG
        letter_filename = os.path.join(output_dir, f"letter_{i}.png")
        success = cv2.imwrite(letter_filename, cell)
        
        if success:
            letter_files.append(letter_filename)
            print(f"[INFO] Saved letter {i} ({w}x{h}) at {letter_filename}")
        else:
            print(f"[WARNING] Failed to save letter {i}")

    if not letter_files:
        print("[WARNING] No letters were extracted! Using fallback approach...")
        # Create at least one letter as fallback
        blank = np.ones((50, 50), dtype=np.uint8) * 255
        circle = cv2.circle(blank.copy(), (25, 25), 20, 0, -1)
        fallback_path = os.path.join(output_dir, "letter_0.png")
        cv2.imwrite(fallback_path, circle)
        letter_files.append(fallback_path)

    # Alternative approach: try to detect grid lines and extract cells
    if len(letter_files) < 10:
        print("[INFO] Trying alternative grid detection approach...")
        grid_letters = extract_letters_from_grid(image_path, output_dir)

        if grid_letters:
            letter_files = grid_letters

    print(f"[SUCCESS] Extracted {len(letter_files)} letters! ✅")
    return letter_files

def extract_letters_from_grid(image_path, output_dir):
    """Alternative approach to extract letters using explicit grid detection."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return []
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Define a character mapping for lowercase and uppercase letters
    char_mapping = {}
    for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        char_mapping[i] = char
    
    # Detect horizontal and vertical lines to find grid
    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)
    
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    # Specify size on vertical axis
    rows = vertical.shape[0]
    vertical_size = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    # Find contours in the grid lines
    h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract horizontal and vertical line positions
    h_lines = []
    v_lines = []
    
    for contour in h_contours:
        x, y, w, h = cv2.boundingRect(contour)
        h_lines.append(y)
    
    for contour in v_contours:
        x, y, w, h = cv2.boundingRect(contour)
        v_lines.append(x)
    
    # Sort lines positions
    h_lines.sort()
    v_lines.sort()
    
    # Remove duplicate lines (those that are very close)
    h_lines = [h_lines[i] for i in range(len(h_lines)) if i == 0 or h_lines[i] - h_lines[i-1] > 10]
    v_lines = [v_lines[i] for i in range(len(v_lines)) if i == 0 or v_lines[i] - v_lines[i-1] > 10]
    
    letter_files = []
    letter_count = 0
    
    # Process each cell in the grid
    for i in range(len(h_lines)-1):
        for j in range(len(v_lines)-1):
            cell_x = v_lines[j]
            cell_y = h_lines[i]
            cell_width = v_lines[j+1] - v_lines[j]
            cell_height = h_lines[i+1] - h_lines[i]
            
            # Extract cell content with margin
            margin = 5
            cell = thresh[
                cell_y + margin:cell_y + cell_height - margin, 
                cell_x + margin:cell_x + cell_width - margin
            ]
            
            # Skip empty cells
            if cell.size == 0 or np.sum(cell) < 500:
                continue
                
            # Assign the correct character based on grid position
            if letter_count < len(char_mapping):
                char = char_mapping[letter_count]
            else:
                char = f"_{letter_count}"
            
            # Save cell as a letter PNG
            letter_filename = os.path.join(output_dir, f"letter_{ord(char)}.png")
            
            # Save letter image
            if cv2.imwrite(letter_filename, cell):
                letter_files.append(letter_filename)
                print(f"[INFO] Saved grid cell letter '{char}' at {letter_filename}")
            
            letter_count += 1
    
    return letter_files

def convert_png_to_svg(input_dir, output_dir):
    """Convert extracted PNG letters into SVG vector files."""
    print("[INFO] 🖼 Converting letters into vector SVGs...")
    
    # Check if potrace is installed
    try:
        subprocess.run(["potrace", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARNING] ⚠️ Potrace tool not found. Please install potrace for SVG conversion.")
        # Return without failing - we'll use PNG files instead
        return
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            try:
                img_path = os.path.join(input_dir, filename)
                pbm_path = img_path.replace(".png", ".pbm")
                svg_path = os.path.join(output_dir, filename.replace(".png", ".svg"))

                # Open image and ensure it's binary (black and white)
                img = Image.open(img_path).convert("1")  # Convert to binary
                
                # Save as PBM with proper format
                img.save(pbm_path)
                
                # Run potrace to convert PBM to SVG
                result = subprocess.run(
                    ["potrace", pbm_path, "-s", "-o", svg_path], 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                # Print debugging info if needed
                if result.stderr:
                    print(f"[INFO] Potrace message: {result.stderr.decode()}")
                    
                # Clean up PBM file
                os.remove(pbm_path)
                
            except Exception as e:
                print(f"[WARNING] ⚠️ Failed to convert {filename} to SVG: {e}")
                # Continue with other files
                continue

    print("[SUCCESS] ✅ SVG conversion completed!")

def fine_tune_model(image, text):
    """Fine-tune the OCR model on the user's handwriting"""
    if not model_loaded or not text.strip():
        print("[INFO] Skipping fine-tuning (model not loaded or text empty)")
        return {"status": "skipped"}

    print("[INFO] Fine-tuning model...")
    try:
      optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
      inputs = processor(image, return_tensors="pt")
      labels = processor.tokenizer(text, return_tensors="pt").input_ids

      # # Ensure `decoder_start_token_id` is set
      # if model.config.decoder_start_token_id is None:
      #     model.config.decoder_start_token_id = (
      #         processor.tokenizer.cls_token_id 
      #         if hasattr(processor.tokenizer, "cls_token_id") 
      #         else processor.tokenizer.bos_token_id  # Use `bos_token_id` as fallback
      #     )

      # # Ensure `pad_token_id` is set
      # if model.config.pad_token_id is None:
      #     if hasattr(processor.tokenizer, "pad_token_id") and processor.tokenizer.pad_token_id is not None:
      #         model.config.pad_token_id = processor.tokenizer.pad_token_id
      #     else:
      #         print("[WARNING] ⚠️ `pad_token_id` is missing! Fine-tuning may not work correctly.")

      # Compute loss and backpropagation
      
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
        print(f"[WARNING] Fine-tuning error: {e}")
        return {"status": "error", "error": str(e)}

def generate_font(text, letters_dir):
    """Generate a font from SVG letters with fallback to PNG if needed"""
    print("[INFO] Generating font...")

    # Create a unique font name
    font_name = f"handwriting_{uuid.uuid4().hex[:8]}"
    script_path = f"generate_font_{font_name}.pe"
    font_path = os.path.join("fonts", f"{font_name}.ttf")

    # Make sure the fonts directory exists
    os.makedirs("fonts", exist_ok=True)

    # Check if FontForge is installed
    try:
        subprocess.run(["fontforge", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARNING] ⚠️ FontForge not found. Returning directory of letter images instead.")
        return {"status": "no_fontforge", "letter_dir": letters_dir}

    # Count available letters
    svg_letters = [f for f in os.listdir(letters_dir) if f.endswith(".svg")]
    png_letters = [f for f in os.listdir(letters_dir) if f.endswith(".png")]
    
    print(f"[INFO] Found {len(svg_letters)} SVG and {len(png_letters)} PNG letters")
    
    # Create FontForge script
    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")
        f.write(f"SetFontNames('{font_name}', 'Handwriting', 'Regular');\n")
        
        # Try to map characters to files
        chars_mapped = 0

        # First check if we have direct character-named files (from grid extraction)
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for char in alphabet:
            char_code = ord(char)
            
            # Try character-specific files first (from grid extraction)
            svg_file = os.path.join(letters_dir, f"letter_{char_code}.svg")
            png_file = os.path.join(letters_dir, f"letter_{char_code}.png")
            
            if os.path.exists(svg_file):
                f.write(f'Select({char_code});\n')
                f.write(f'Import("{svg_file}");\n')
                chars_mapped += 1
                print(f"[INFO] Mapped character '{char}' to {svg_file}")
            elif os.path.exists(png_file):
                f.write(f'Select({char_code});\n')
                f.write(f'Import("{png_file}");\n')
                chars_mapped += 1
                print(f"[INFO] Mapped character '{char}' to {png_file}")
        
        # If we don't have any character-named files, fall back to numbered files
        if chars_mapped == 0:
            for i, char in enumerate(text):
                if i >= 52:  # Only use the first 52 characters (a-zA-Z)
                    break
                    
                char_code = ord(char)
                
                # Try SVG first (preferred)
                svg_file = os.path.join(letters_dir, f"letter_{i}.svg")
                png_file = os.path.join(letters_dir, f"letter_{i}.png")
                
                if os.path.exists(svg_file):
                    f.write(f'Select({char_code});\n')
                    f.write(f'Import("{svg_file}");\n')
                    chars_mapped += 1
                    print(f"[INFO] Mapped character '{char}' to {svg_file}")
                elif os.path.exists(png_file):
                    f.write(f'Select({char_code});\n')
                    f.write(f'Import("{png_file}");\n')
                    chars_mapped += 1
                    print(f"[INFO] Mapped character '{char}' to {png_file}")

        # Generate if we have any characters mapped
        if chars_mapped > 0:
            f.write(f'Generate("{font_path}");\n')
        f.write('Quit(0);\n')
    
    # Run FontForge script
    try:
        result = subprocess.run(
            ["fontforge", "-script", script_path], 
            check=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Log output from FontForge for debugging
        if result.stdout:
            print(f"[INFO] FontForge output: {result.stdout.decode()}")
        
        # Clean up
        os.remove(script_path)
        
        if os.path.exists(font_path):
            print(f"[SUCCESS] Font generated at {font_path}!")
            return font_path
        else:
            print("[WARNING] Font file was not created")
            return {"status": "font_not_created", "letter_dir": letters_dir}
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FontForge error: {e}")
        if e.stderr:
            print(f"Details: {e.stderr.decode()}")
        return {"status": "fontforge_error", "error": str(e), "letter_dir": letters_dir}