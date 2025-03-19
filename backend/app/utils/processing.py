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
    # Define the expected alphabet order
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Sort letter files by their index to ensure correct order
    sorted_files = sorted(letter_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))

    if not model_loaded:
        print("[INFO] OCR model not loaded, using default alphabet mapping")
        return list(alphabet[:min(len(sorted_files), len(alphabet))])
    
    recognized_chars = []
    for i, file_path in enumerate(sorted_files):
        if i >= len(alphabet):
            break
            
        try:
            image = Image.open(file_path).convert("RGB") 
            # Try OCR first
            text = ocr_model(image)
            
            # If OCR returns garbage or multiple chars, use expected alphabet instead
            if (not text or len(text.strip()) != 1 or 
                text.strip() not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                text = alphabet[i]
                print(f"[INFO] Replacing OCR result '{text}' with expected '{alphabet[i]}' for {file_path}")
            
            recognized_chars.append(text.strip())
        except Exception as e:
            print(f"[WARNING] OCR failed for letter {i}: {e}")
            # Fallback to expected alphabet
            if i < len(alphabet):
                recognized_chars.append(alphabet[i])
            else:
                recognized_chars.append("?")
    
    print(f"[INFO] Recognized {len(recognized_chars)} characters")
    return recognized_chars

# Letter Extrations Functions
def extract_letters(image_path, output_dir):
    """Extract individual letters from handwriting image (without a grid)."""
    print("[INFO] ✂️ Extracting letters from handwriting image...")

    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply threshold to make text stand out (adjust values if needed)
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours (shapes) in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right, top to bottom
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 50, b[0]))  # Sort by row, then column

    extracted_letters = []
    os.makedirs(output_dir, exist_ok=True)

    # Loop through detected letters and save them
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        if w < 5 or h < 5:  # Skip tiny noise elements
            continue
        
        # Extract the letter from the image
        letter_image = image[y:y+h, x:x+w]

        # Save as PNG
        letter_path = os.path.join(output_dir, f"letter_{i}.png")
        cv2.imwrite(letter_path, letter_image)
        extracted_letters.append(letter_path)

    print(f"[SUCCESS] ✅ Extracted {len(extracted_letters)} letters!")
    return extracted_letters

def convert_png_to_svg(input_dir, output_dir):
    """Convert extracted PNG letters into SVG vector files."""
    import os
    import subprocess
    from PIL import Image

    print("[INFO] 🖼 Converting letters into vector SVGs...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if potrace is installed
    try:
        subprocess.run(["potrace", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARNING] ⚠️ Potrace tool not found. Please install potrace for SVG conversion.")
        # Return without failing - we'll use PNG files instead
        return
    
    conversion_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            try:
                img_path = os.path.join(input_dir, filename)
                pbm_path = os.path.join(output_dir, filename.replace(".png", ".pbm"))
                svg_path = os.path.join(output_dir, filename.replace(".png", ".svg"))

                # Skip if SVG already exists
                if os.path.exists(svg_path):
                    continue
                
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
                
                conversion_count += 1
                
                # Clean up PBM file
                if os.path.exists(pbm_path):
                    os.remove(pbm_path)
                    
            except Exception as e:
                print(f"[WARNING] ⚠️ Failed to convert {filename} to SVG: {e}")
                # Continue with other files
                continue

    print(f"[SUCCESS] ✅ SVG conversion completed! Converted {conversion_count} files.")

def fine_tune_model(images, texts):
    """Fine-tune the OCR model on the user's handwriting"""
    if not model_loaded or not texts:
        print("[INFO] Skipping fine-tuning (model not loaded or text empty)")
        return {"status": "skipped"}

    print("[INFO] Fine-tuning model...")

    try:
        # Fix decoder_start_token_id issue
        if not hasattr(model.config, "decoder_start_token_id") or model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = 2  # This is the default value for the TrOCR model
        
        # Fix pad_token_id issue
        model.config.pad_token_id = 1  # Set pad token ID explicitly
        
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