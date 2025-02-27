import os
import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance
import subprocess

# Try to load OCR (pre-trained) OCR model
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("[SUCCESS] 🤖AI model loaded!")
    model_loaded = True
except Exception as e:
    print(f"[WARNING] ⚠️ Could not load AI model: {e}")
    print("[INFO] Will use fallback alphabet instead of OCR")
    model_loaded = False

def ocr_model(image):
    """Perform OCR on an image using TrOCR"""
    try: 
        inputs = processor(image, return_tensors="pt")
        generated_text = model.generate(**inputs)
        return processor.batch_decode(generated_text, skip_special_tokens=True)[0].strip()
    except Exception as e: 
        print(f"[ERROR] ❌ OCR model error: {e}")
        return "?" # Return "?" if OCR fails

# Preprocessing the image for OCR Model
def process_image(image_path): 
    """Preprocess the image for OCR"""
    try: 
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.grayscale(image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3.0)
        return image.convert("RGB")  # Ensure 3-channel image for OCR
    except Exception as e:
        print(f"[ERROR] ❌ Image processing error: {e}")
        return None # Return None if processing fails

# OCR Model recognizing handwriting image to alphabet
def recognize_handwriting(letter_files):
    """Perform OCR on extracted letter images and map them to characters"""
    if not model_loaded:
        return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    recognized_chars = []  # This should be the list used
    for letter_file in letter_files:
        try:
            processed_image = process_image(letter_file) 
            if processed_image:                 
                text = ocr_model(processed_image)  # Assuming ocr_model() expects an image object
                recognized_chars.append(text)  # ✅ Append to recognized_chars instead
            else: 
                recognized_chars.append("?") # Handle failed processing
        except Exception as e:
                print(f"[WARNING] OCR failed for a letter: {e}")
                recognized_chars.append("?")  # Use '?' for unrecognized letters
    
    return recognized_chars

# Letter Extrations Functions
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
    print("[INFO] Trying improved grid detection approach...")
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return []
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a slight blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold more aggressively to separate content from grid
    _, thresh = cv2.threshold(gray_blurred, 180, 255, cv2.THRESH_BINARY_INV)
    
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

    # Create debug directory
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Calculate expected number of characters 
    expected_chars = 52  # a-z, A-Z

    # Process each cell in the grid, but limit to the expected number
    for i in range(len(h_lines)-1):
        for j in range(len(v_lines)-1):
            # Stop if we've already extracted the expected number of characters
            if letter_count >= expected_chars:
                break
            cell_x = v_lines[j]
            cell_y = h_lines[i]
            cell_width = v_lines[j+1] - v_lines[j]
            cell_height = h_lines[i+1] - h_lines[i]
            
            # Extract cell content with larger margin to avoid grid lines
            margin = 10
            
            # Ensure margins are within image bounds
            y_start = max(0, cell_y + margin)
            y_end = min(gray.shape[0], cell_y + cell_height - margin)
            x_start = max(0, cell_x + margin)
            x_end = min(gray.shape[1], cell_x + cell_width - margin)

            # Skip if the cell is too small
            if (y_end - y_start) < 10 or (x_end - x_start) < 10:
                continue
            
            # Extract the cell region from the original grayscale image
            cell_gray = gray[y_start:y_end, x_start:x_end]
            
            # Skip if cell is too small
            if cell_gray.size == 0:
                continue
            
            # Apply adaptive thresholding to better handle varying lighting
            cell_binary = cv2.adaptiveThreshold(
                cell_gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                15, 
                9
            )
            
            # Apply morphological operations to remove noise and thin lines
            kernel = np.ones((2, 2), np.uint8)
            cell_cleaned = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)
            
            # Dilate to make the letters more prominent
            cell_dilated = cv2.dilate(cell_cleaned, kernel, iterations=1)
            
            # Skip empty cells (no significant black pixels)
            if np.sum(cell_dilated) < 500:
                continue

            # IMPORTANT: Invert the image for proper font rendering
            # FontForge expects white letters on black background
            cell_inverted = cv2.bitwise_not(cell_dilated)
            
            # Only process the expected number of characters (a-z, A-Z)
            if letter_count < len(char_mapping):
                char = char_mapping[letter_count]
                char_code = ord(char)
                
                # Save processed cell as a letter PNG
                letter_filename = os.path.join(output_dir, f"letter_{char_code}.png")
                
                # Save letter image
                if cv2.imwrite(letter_filename, cell_inverted):
                    letter_files.append(letter_filename)
                    print(f"[INFO] Saved grid cell letter '{char}' at {letter_filename}")
                    
                    # Save debug images
                    cv2.imwrite(os.path.join(debug_dir, f"gray_{char}.png"), cell_gray)
                    cv2.imwrite(os.path.join(debug_dir, f"binary_{char}.png"), cell_binary)
                    cv2.imwrite(os.path.join(debug_dir, f"final_{char}.png"), cell_inverted)
            
                    letter_count += 1
    
        # Break out of nested loop if we've extracted all expected characters
        if letter_count >= expected_chars:
            break

    print(f"[INFO] Grid extraction extracted {len(letter_files)} letters")
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

def fine_tune_model(images, texts):
    """Fine-tune the OCR model on the user's handwriting"""
    if not model_loaded or not texts:
        print("[INFO] Skipping fine-tuning (model not loaded or text empty)")
        return {"status": "skipped"}

    print("[INFO] Fine-tuning model...")

    try:
        # Step 1: Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

        for img, text in zip(images, texts):
            inputs = processor(img, return_tensors="pt")
            labels = processor.tokenizer(text, return_tensors="pt").input_ids

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

        stats = {
            "status": "success",
            "loss": float(loss.item()),
            "text_length": len(text),
            "unique_chars": len(set(text))
        }

        print(f"[SUCCESS] 🏆 Fine-tuning completed successfully! Loss: {loss.item():.4f}")
        return {"stats": stats}

    except Exception as e:
        # Free up memory in case of failure
        with torch.no_grad():
            print(f"[WARNING] Fine-tuning error: {e}")
        return {"status": "error", "error": str(e)}
