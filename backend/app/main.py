from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse  # ✅ Add this line!
from app.routes import router

app = FastAPI(title="Handwriting to Font Converter")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")

app.include_router(router)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>Welcome to the Handwriting to Font Converter API! 🚀</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# """
# Handwriting to Font Converter Workshop App
# ------------------------------------------
# A FastAPI application that converts handwritten characters to a font.
# """
# import os
# import uuid
# import shutil
# import subprocess
# import torch
# import cv2
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, Form, Request
# from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image, ImageOps, ImageEnhance

# # Initialize FastAPI app
# app = FastAPI(title="Handwriting to Font Converter")

# # ==================== SETUP DIRECTORIES & STATIC FILES ====================

# # Set up static files and templates (will create if they don't exist)
# for directory in ["static", "templates", "uploads", "fonts"]:
#     os.makedirs(directory, exist_ok=True)

# # Mount static directories
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
# app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")

# print("🐲[INFO] Directories created: uploads/, fonts/, templates/, static/")

# # ==================== WEB ROUTES ====================

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     print("[INFO] 🌳 Root endpoint accessed")
#     try:
#         return templates.TemplateResponse("index.html", {"request": request})
#     except Exception as e:
#         # Fallback HTML with improved UI
#         return HTMLResponse(content="""
#         <html>
#             <head>
#                 <title>Handwriting to Font Converter</title>
#                 <style>
#                     body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
#                     h1, h2 { color: #333; }
#                     .container { max-width: 800px; margin: 0 auto; padding: 20px; }
#                     .upload-form { background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
#                     .btn { 
#                         background-color: #4CAF50; 
#                         color: white; 
#                         padding: 10px 15px; 
#                         border: none; 
#                         border-radius: 4px; 
#                         cursor: pointer; 
#                         text-decoration: none;
#                         display: inline-block;
#                         margin-top: 10px;
#                     }
#                     .btn:hover { background-color: #45a049; }
#                     #result { display: none; margin-top: 20px; padding: 20px; background-color: #f0f0f0; border-radius: 8px; }
#                     #preview { margin-top: 20px; font-size: 24px; }
#                     .loader { 
#                         border: 5px solid #f3f3f3;
#                         border-top: 5px solid #3498db;
#                         border-radius: 50%;
#                         width: 30px;
#                         height: 30px;
#                         animation: spin 2s linear infinite;
#                         display: none;
#                         margin: 20px auto;
#                     }
#                     @keyframes spin {
#                         0% { transform: rotate(0deg); }
#                         100% { transform: rotate(360deg); }
#                     }
#                 </style>
#             </head>
#             <body>
#                 <div class="container">
#                     <h1>Handwriting to Font Converter</h1>
                    
#                     <div class="upload-form">
#                         <h2>Step 1: Get the Template</h2>
#                         <p>Download our <a href="/template">handwriting template</a> and fill it in with your handwriting.</p>
                        
#                         <h2>Step 2: Upload Your Handwriting</h2>
#                         <form id="uploadForm">
#                             <input type="file" name="file" accept="image/*" required>
#                             <button type="submit" class="btn">Create My Font</button>
#                         </form>
#                         <div class="loader" id="loader"></div>
#                     </div>
                    
#                     <div id="result">
#                         <h2>Your Font is Ready!</h2>
#                         <p>Your handwriting has been converted to a font.</p>
#                         <a id="downloadLink" href="#" class="btn">Download Font</a>
#                         <a id="debugLink" href="#" target="_blank" style="margin-left: 10px;">View Extracted Letters</a>
                        
#                         <div id="preview">
#                             <h3>Preview:</h3>
#                             <p id="previewText">The quick brown fox jumps over the lazy dog.</p>
#                         </div>
#                     </div>
#                 </div>
                
#                 <script>
#                     document.getElementById('uploadForm').addEventListener('submit', async (e) => {
#                         e.preventDefault();
                        
#                         // Show loader
#                         document.getElementById('loader').style.display = 'block';
                        
#                         // Hide previous result if any
#                         document.getElementById('result').style.display = 'none';
                        
#                         const formData = new FormData(e.target);
                        
#                         try {
#                             const response = await fetch('/upload', {
#                                 method: 'POST',
#                                 body: formData
#                             });
                            
#                             const data = await response.json();
                            
#                             // Hide loader
#                             document.getElementById('loader').style.display = 'none';
                            
#                             if (data.success) {
#                                 // Show result section
#                                 document.getElementById('result').style.display = 'block';
                                
#                                 // Set download link
#                                 const downloadLink = document.getElementById('downloadLink');
#                                 downloadLink.href = data.download_url;
#                                 downloadLink.download = data.font_url.split('/').pop();
                                
#                                 // Set debug link
#                                 const debugLink = document.getElementById('debugLink');
#                                 debugLink.href = data.debug_url;
#                                 debugLink.textContent = 'View Extracted Letters';
                                
#                                 // Apply the font to preview text
#                                 const previewText = document.getElementById('previewText');
                                
#                                 // Create a style element to load the font
#                                 const style = document.createElement('style');
#                                 style.textContent = `
#                                     @font-face {
#                                         font-family: 'YourHandwriting';
#                                         src: url('${data.download_url}') format('truetype');
#                                     }
#                                     #previewText {
#                                         font-family: 'YourHandwriting', sans-serif;
#                                     }
#                                 `;
#                                 document.head.appendChild(style);
#                             } else {
#                                 alert('Error: ' + (data.error || 'Unknown error'));
#                             }
#                         } catch (error) {
#                             // Hide loader
#                             document.getElementById('loader').style.display = 'none';
                            
#                             console.error('Error:', error);
#                             alert('Error: ' + error.message);
#                         }
#                     });
#                 </script>
#             </body>
#         </html>
#         """)

# @app.get("/template")
# async def get_template():
#     """Provide a template for handwriting"""
#     template_path = "static/handwriting_template.png"
    
#     # Create template if it doesn't exist
#     if not os.path.exists(template_path):
#         create_template(template_path)
    
#     return FileResponse(template_path)

# def cleanup_old_sessions(max_sessions=10):
#     """Clean up old upload sessions to prevent accumulation of files"""
#     uploads_dir = "uploads"
#     if not os.path.exists(uploads_dir):
#         return
        
#     # List all session directories
#     sessions = []
#     for item in os.listdir(uploads_dir):
#         item_path = os.path.join(uploads_dir, item)
#         if os.path.isdir(item_path):
#             # Get creation time
#             created_time = os.path.getctime(item_path)
#             sessions.append((item, created_time))
    
#     # Sort by creation time (newest first)
#     sessions.sort(key=lambda x: x[1], reverse=True)
    
#     # Keep only the most recent sessions
#     if len(sessions) > max_sessions:
#         for session, _ in sessions[max_sessions:]:
#             session_path = os.path.join(uploads_dir, session)
#             try:
#                 shutil.rmtree(session_path)
#                 print(f"[INFO] Cleaned up old session: {session}")
#             except Exception as e:
#                 print(f"[WARNING] Failed to remove session {session}: {e}")

# @app.post("/upload")
# async def upload_image(file: UploadFile = File(...)):
#     # Run cleanup before processing new upload
#     cleanup_old_sessions()

#     """Process uploaded handwriting image and convert to font"""
#     print(f"[INFO] Received file ✍️📝: {file.filename}")
    
#     # Create a unique ID for this user session
#     session_id = str(uuid.uuid4())
#     print(f"[INFO] 🧘 Generated session ID: {session_id}")
    
#     # Create unique directories for this user's processing
#     user_upload_dir = f"uploads/{session_id}"
#     user_letters_dir = f"uploads/{session_id}/letters"
#     os.makedirs(user_upload_dir, exist_ok=True)
#     os.makedirs(user_letters_dir, exist_ok=True)
#     print(f"[INFO] Created user directories: {user_upload_dir}, {user_letters_dir}")

#     # Save the uploaded file
#     file_path = os.path.join(user_upload_dir, file.filename)
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         print(f"[SUCCESS] File saved at {file_path}")
#     except Exception as e:
#         print(f"[ERROR] 👹 Failed to save file: {e}")
#         return JSONResponse(content={"error": "Failed to save file"}, status_code=500)

#     # Step 1: Process Image with AI Model
#     try:
#         # Process the image
#         processed_image = process_image(file_path)

#         # Recognize text if model is loaded
#         if model_loaded: 
#             recognized_text = recognize_handwriting(processed_image)
#             print(f"[SUCCESS] Recognized text: {recognized_text}")
#         else:
#             # Use fallback if model not loaded
#             recognized_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#             print(f"[INFO] Using fallback text: {recognized_text}")

#         # If OCR text is too short, use a fallback alphabet    
#         if not recognized_text or len(recognized_text) < 10:
#             print("[WARNING] ⚠️ OCR produced poor results. Using fallback alphabet.")
#             recognized_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#             print(f"[INFO] Using fallback text: {recognized_text}")

#         # Extract handwritten letters
#         letter_images = extract_letters(file_path, user_letters_dir)
#         if not letter_images: 
#             raise ValueError("No letters were extracted!")
        
#         # Attempt to convert PNG to SVG (but continue even if it fails)
#         try:
#             convert_png_to_svg(user_letters_dir, user_letters_dir)
#         except Exception as e:
#             print(f"[WARNING] SVG conversion issue: {e}")
#             print("[INFO] Will continue with PNG files only")
        
#     except Exception as e:
#         print(f"[ERROR] Image processing failed: {e}")
#         return JSONResponse(content={"error": f"Image processing failed: {str(e)}"}, status_code=500)
    
#     # Step 2: Fine-Tune Model (optional)
#     if model_loaded:
#         try:
#             fine_tune_stats = fine_tune_model(processed_image, recognized_text)
#             print("[SUCCESS] Fine-tuning complete 🏆")
#             print(f"[INFO] Fine-tuning stats: {fine_tune_stats}")
#         except Exception as e:
#             print(f"[WARNING] 👹 Fine-tuning issue: {e}")
#             # Continue anyway, as font generation can still work

#     # Step 3: Generate Font
#     try:
#         font_result = generate_font(recognized_text, user_letters_dir)

#         # Check font generation result
#         if isinstance(font_result, str) and os.path.exists(font_result):
#             print(f"[SUCCESS] Font generated: {font_result}")
#             font_url = os.path.join("fonts", os.path.basename(font_result))
#             success = True
#         else:
#             print(f"[WARNING] Font generation produced alternative result: {font_result}")
#             font_url = None
#             success = False

#         # Return result
#         return {
#             "success": success,
#             "recognized_text": recognized_text,
#             "font_url": font_url,
#             "download_url": f"/download-font/{os.path.basename(font_result)}" if success else None,
#             "session_id": session_id,
#             "letters_dir": user_letters_dir, 
#             "debug_url": f"/debug/{session_id}"
#         }
    
#     except Exception as e:
#         print(f"[ERROR] 👹 Font generation failed: {e}")
#         return JSONResponse(
#             content={
#                 "error": f"Font generation failed: {str(e)}",
#                 "session_id": session_id,
#                 "letters_dir": user_letters_dir
#             },
#             status_code=500
#         )
    
# @app.get("/download-font/{font_name}")
# async def download_font(font_name: str):
#     """Direct font download endpoint"""
#     font_path = os.path.join("fonts", font_name)
    
#     if not os.path.exists(font_path):
#         return JSONResponse(content={"error": "Font not found"}, status_code=404)
    
#     return FileResponse(
#         path=font_path, 
#         filename=font_name,
#         media_type="font/ttf"
#     )


# @app.get("/debug/{session_id}")
# async def debug_letters(request: Request, session_id: str):
#     """Debug endpoint to view extracted letters"""
#     letters_dir = f"uploads/{session_id}/letters"
    
#     if not os.path.exists(letters_dir):
#         return HTMLResponse(content="Session not found or letters not extracted.")
    
#     # Find all letter files
#     file_infos = []
#     for filename in sorted(os.listdir(letters_dir)):
#         if filename.startswith("letter_") and (filename.endswith(".png") or filename.endswith(".svg")):
#             try:
#                 code_part = filename.split("_")[1].split(".")[0]
#                 if code_part.isdigit():
#                     code = int(code_part)
#                     try:
#                         char = chr(code)
#                         file_infos.append({
#                             "path": f"/uploads/{session_id}/letters/{filename}",
#                             "code": code,
#                             "char": char,
#                             "filename": filename
#                         })
#                     except ValueError:
#                         file_infos.append({
#                             "path": f"/uploads/{session_id}/letters/{filename}", 
#                             "code": code,
#                             "char": "?",
#                             "filename": filename
#                         })
#             except (IndexError, ValueError):
#                 pass
    
#     # Sort by character code
#     file_infos.sort(key=lambda x: x["code"])
    
#     html_content = f"""
#     <html>
#         <head>
#             <title>Letter Extraction Debug</title>
#             <style>
#                 body {{ font-family: Arial, sans-serif; margin: 20px; }}
#                 .letter-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }}
#                 .letter-cell {{ 
#                     border: 1px solid #ccc; 
#                     padding: 10px; 
#                     display: flex; 
#                     flex-direction: column; 
#                     align-items: center;
#                     background-color: #f9f9f9;
#                 }}
#                 .letter-image {{ 
#                     width: 100px; 
#                     height: 100px; 
#                     background-color: white;
#                     display: flex;
#                     align-items: center;
#                     justify-content: center;
#                 }}
#                 .letter-info {{ text-align: center; margin-top: 5px; }}
#                 h1, h2 {{ color: #333; }}
#             </style>
#         </head>
#         <body>
#             <h1>Letter Extraction Debug</h1>
#             <p>Session ID: {session_id}</p>
#             <h2>Extracted Letters ({len(file_infos)})</h2>
#             <div class="letter-grid">
#     """
    
#     for info in file_infos:
#         html_content += f"""
#                 <div class="letter-cell">
#                     <div class="letter-image">
#                         <img src="{info['path']}" style="max-width: 100%; max-height: 100%;">
#                     </div>
#                     <div class="letter-info">
#                         <strong>'{info['char']}'</strong> (Code: {info['code']})
#                     </div>
#                 </div>
#         """
    
#     html_content += """
#             </div>
#         </body>
#     </html>
#     """
    
#     return HTMLResponse(content=html_content)


# # ==================== TEMPLATE GENERATION ====================

# def create_template(path):
#     """Create a template for handwriting with clear cell boundaries"""
#     print("🐲🐲🐲🐲🐲")
#     # Set the size of the template
#     width, height = 1200, 1800
#     margin = 50
#     cell_size = 120
    
#     # Create a white background
#     img = Image.new('RGB', (width, height), color='white')
#     draw = ImageDraw.Draw(img)
    
#     # Calculate grid dimensions
#     cols = 6
#     rows = 10  # Enough for a-z, A-Z
    
#     # Draw the grid with labels
#     chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
#     # Use a font for cell labels
#     try:
#         font = ImageFont.truetype("arial.ttf", 14)
#     except IOError:
#         font = ImageFont.load_default()
    
#     # Draw the grid
#     char_index = 0
#     for row in range(rows):
#         for col in range(cols):
#             if char_index >= len(chars):
#                 break
                
#             # Calculate cell position
#             x1 = margin + col * cell_size
#             y1 = margin + row * cell_size
#             x2 = x1 + cell_size
#             y2 = y1 + cell_size
            
#             # Draw cell border (thicker for clarity)
#             draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            
#             # Draw character label in top-left corner
#             label = chars[char_index]
#             draw.text((x1 + 5, y1 + 5), label, fill='black', font=font)
            
#             char_index += 1
    
#     # Add instructions at the top
#     instructions = "Instructions: Draw each letter inside its cell. Keep your handwriting within the boundaries."
#     draw.text((margin, 20), instructions, fill='black', font=font)
    
#     # Save the template
#     img.save(path)
#     print(f"[INFO] Created improved template at {path}")
    
#     return path



# def extract_letters(image_path, output_dir):
#     """Extract individual handwritten letters from a grid template."""    
#     print("[INFO] ✂️ Extracting letters from handwriting grid...")

#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Failed to load image: {image_path}")
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
#     # Apply adaptive thresholding to better handle grid lines vs. text
#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                   cv2.THRESH_BINARY_INV, 11, 2)
#     # Debug info
#     print(f"[INFO] Image shape: {image.shape}, unique pixel values: {np.unique(binary)}")
    
#     # Try to detect grid cells - we'll look for rectangles
#     # First we need to remove some noise with morphological operations
#     kernel = np.ones((2,2), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

#     # Find contours - these should be our grid cells and letters
#     contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print(f"[INFO] Found {len(contours)} potential contours")

#     # Filter and sort contours by position (top to bottom, left to right)
#     valid_contours = []
#     min_contour_area = 100  # Adjust based on your image resolution
    
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         area = w * h
        
#         # Skip very small contours (noise)
#         if area < min_contour_area:
#             continue
            
#         valid_contours.append((x, y, w, h, contour))
    
#     # Sort contours by rows, then by columns
#     # This assumes the grid cells are arranged in rows
#     row_sorted = sorted(valid_contours, key=lambda c: c[1])

#     # Make sure output directory exists 
#     os.makedirs(output_dir, exist_ok=True)

#     letter_files = []

#     # Process each contour as a potential cell with a letter
#     for i, (x, y, w, h, contour) in enumerate(row_sorted):
#         # Extract region containing the letter
#         cell = binary[y:y+h, x:x+w]
        
#         # Skip empty cells (no significant black pixels)
#         if np.sum(cell) < 500:  # Threshold for "significant" content
#             print(f"[INFO] Skipping empty cell at ({x}, {y})")
#             continue
            
#         # Save as PNG
#         letter_filename = os.path.join(output_dir, f"letter_{i}.png")
#         success = cv2.imwrite(letter_filename, cell)
        
#         if success:
#             letter_files.append(letter_filename)
#             print(f"[INFO] Saved letter {i} ({w}x{h}) at {letter_filename}")
#         else:
#             print(f"[WARNING] Failed to save letter {i}")

#     if not letter_files:
#         print("[WARNING] No letters were extracted! Using fallback approach...")
#         # Create at least one letter as fallback
#         blank = np.ones((50, 50), dtype=np.uint8) * 255
#         circle = cv2.circle(blank.copy(), (25, 25), 20, 0, -1)
#         fallback_path = os.path.join(output_dir, "letter_0.png")
#         cv2.imwrite(fallback_path, circle)
#         letter_files.append(fallback_path)

#     # Alternative approach: try to detect grid lines and extract cells
#     if len(letter_files) < 10:
#         print("[INFO] Trying alternative grid detection approach...")
#         grid_letters = extract_letters_from_grid(image_path, output_dir)

#         if grid_letters:
#             letter_files = grid_letters

#     print(f"[SUCCESS] Extracted {len(letter_files)} letters! ✅")
#     return letter_files

# def extract_letters_from_grid(image_path, output_dir):
#     """Alternative approach to extract letters using explicit grid detection."""
#     print("[INFO] Trying improved grid detection approach...")
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         return []
        
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply a slight blur to reduce noise
#     gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Threshold more aggressively to separate content from grid
#     _, thresh = cv2.threshold(gray_blurred, 180, 255, cv2.THRESH_BINARY_INV)
    
#     # Define a character mapping for lowercase and uppercase letters
#     char_mapping = {}
#     for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
#         char_mapping[i] = char
    
#     # Detect horizontal and vertical lines to find grid
#     horizontal = np.copy(thresh)
#     vertical = np.copy(thresh)
    
#     # Specify size on horizontal axis
#     cols = horizontal.shape[1]
#     horizontal_size = cols // 30
#     horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
#     horizontal = cv2.erode(horizontal, horizontalStructure)
#     horizontal = cv2.dilate(horizontal, horizontalStructure)
    
#     # Specify size on vertical axis
#     rows = vertical.shape[0]
#     vertical_size = rows // 30
#     verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
#     vertical = cv2.erode(vertical, verticalStructure)
#     vertical = cv2.dilate(vertical, verticalStructure)
    
#     # Find contours in the grid lines
#     h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Extract horizontal and vertical line positions
#     h_lines = []
#     v_lines = []
    
#     for contour in h_contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         h_lines.append(y)
    
#     for contour in v_contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         v_lines.append(x)
    
#     # Sort lines positions
#     h_lines.sort()
#     v_lines.sort()
    
#     # Remove duplicate lines (those that are very close)
#     h_lines = [h_lines[i] for i in range(len(h_lines)) if i == 0 or h_lines[i] - h_lines[i-1] > 10]
#     v_lines = [v_lines[i] for i in range(len(v_lines)) if i == 0 or v_lines[i] - v_lines[i-1] > 10]
    
#     letter_files = []
#     letter_count = 0

#     # Create debug directory
#     debug_dir = os.path.join(output_dir, "debug")
#     os.makedirs(debug_dir, exist_ok=True)
    
#     # Calculate expected number of characters 
#     expected_chars = 52  # a-z, A-Z

#     # Process each cell in the grid, but limit to the expected number
#     for i in range(len(h_lines)-1):
#         for j in range(len(v_lines)-1):
#             # Stop if we've already extracted the expected number of characters
#             if letter_count >= expected_chars:
#                 break
#             cell_x = v_lines[j]
#             cell_y = h_lines[i]
#             cell_width = v_lines[j+1] - v_lines[j]
#             cell_height = h_lines[i+1] - h_lines[i]
            
#             # Extract cell content with larger margin to avoid grid lines
#             margin = 10
            
#             # Ensure margins are within image bounds
#             y_start = max(0, cell_y + margin)
#             y_end = min(gray.shape[0], cell_y + cell_height - margin)
#             x_start = max(0, cell_x + margin)
#             x_end = min(gray.shape[1], cell_x + cell_width - margin)

#             # Skip if the cell is too small
#             if (y_end - y_start) < 10 or (x_end - x_start) < 10:
#                 continue
            
#             # Extract the cell region from the original grayscale image
#             cell_gray = gray[y_start:y_end, x_start:x_end]
            
#             # Skip if cell is too small
#             if cell_gray.size == 0:
#                 continue
            
#             # Apply adaptive thresholding to better handle varying lighting
#             cell_binary = cv2.adaptiveThreshold(
#                 cell_gray, 
#                 255, 
#                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                 cv2.THRESH_BINARY_INV, 
#                 15, 
#                 9
#             )
            
#             # Apply morphological operations to remove noise and thin lines
#             kernel = np.ones((2, 2), np.uint8)
#             cell_cleaned = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)
            
#             # Dilate to make the letters more prominent
#             cell_dilated = cv2.dilate(cell_cleaned, kernel, iterations=1)
            
#             # Skip empty cells (no significant black pixels)
#             if np.sum(cell_dilated) < 500:
#                 continue

#             # IMPORTANT: Invert the image for proper font rendering
#             # FontForge expects white letters on black background
#             cell_inverted = cv2.bitwise_not(cell_dilated)
            
#             # Only process the expected number of characters (a-z, A-Z)
#             if letter_count < len(char_mapping):
#                 char = char_mapping[letter_count]
#                 char_code = ord(char)
                
#                 # Save processed cell as a letter PNG
#                 letter_filename = os.path.join(output_dir, f"letter_{char_code}.png")
                
#                 # Save letter image
#                 if cv2.imwrite(letter_filename, cell_inverted):
#                     letter_files.append(letter_filename)
#                     print(f"[INFO] Saved grid cell letter '{char}' at {letter_filename}")
                    
#                     # Save debug images
#                     cv2.imwrite(os.path.join(debug_dir, f"gray_{char}.png"), cell_gray)
#                     cv2.imwrite(os.path.join(debug_dir, f"binary_{char}.png"), cell_binary)
#                     cv2.imwrite(os.path.join(debug_dir, f"final_{char}.png"), cell_inverted)
            
#                     letter_count += 1
    
#         # Break out of nested loop if we've extracted all expected characters
#         if letter_count >= expected_chars:
#             break

#     print(f"[INFO] Grid extraction extracted {len(letter_files)} letters")
#     return letter_files

# def convert_png_to_svg(input_dir, output_dir):
#     """Convert extracted PNG letters into SVG vector files."""
#     print("[INFO] 🖼 Converting letters into vector SVGs...")
    
#     # Check if potrace is installed
#     try:
#         subprocess.run(["potrace", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
#     except (subprocess.SubprocessError, FileNotFoundError):
#         print("[WARNING] ⚠️ Potrace tool not found. Please install potrace for SVG conversion.")
#         # Return without failing - we'll use PNG files instead
#         return
    
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".png"):
#             try:
#                 img_path = os.path.join(input_dir, filename)
#                 pbm_path = img_path.replace(".png", ".pbm")
#                 svg_path = os.path.join(output_dir, filename.replace(".png", ".svg"))

#                 # Open image and ensure it's binary (black and white)
#                 img = Image.open(img_path).convert("1")  # Convert to binary
                
#                 # Save as PBM with proper format
#                 img.save(pbm_path)
                
#                 # Run potrace to convert PBM to SVG
#                 result = subprocess.run(
#                     ["potrace", pbm_path, "-s", "-o", svg_path], 
#                     check=True, 
#                     stdout=subprocess.PIPE, 
#                     stderr=subprocess.PIPE
#                 )
                
#                 # Print debugging info if needed
#                 if result.stderr:
#                     print(f"[INFO] Potrace message: {result.stderr.decode()}")
                    
#                 # Clean up PBM file
#                 os.remove(pbm_path)
                
#             except Exception as e:
#                 print(f"[WARNING] ⚠️ Failed to convert {filename} to SVG: {e}")
#                 # Continue with other files
#                 continue

#     print("[SUCCESS] ✅ SVG conversion completed!")

# # ==================== FINE TUNING FOR OCR MODEL ====================

# def fine_tune_model(image, text):
#     """Fine-tune the OCR model on the user's handwriting"""
#     if not model_loaded or not text.strip():
#         print("[INFO] Skipping fine-tuning (model not loaded or text empty)")
#         return {"status": "skipped"}

#     print("[INFO] Fine-tuning model...")
#     try:
#       optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
#       inputs = processor(image, return_tensors="pt")
#       labels = processor.tokenizer(text, return_tensors="pt").input_ids

#       # # Ensure `decoder_start_token_id` is set
#       # if model.config.decoder_start_token_id is None:
#       #     model.config.decoder_start_token_id = (
#       #         processor.tokenizer.cls_token_id 
#       #         if hasattr(processor.tokenizer, "cls_token_id") 
#       #         else processor.tokenizer.bos_token_id  # Use `bos_token_id` as fallback
#       #     )

#       # # Ensure `pad_token_id` is set
#       # if model.config.pad_token_id is None:
#       #     if hasattr(processor.tokenizer, "pad_token_id") and processor.tokenizer.pad_token_id is not None:
#       #         model.config.pad_token_id = processor.tokenizer.pad_token_id
#       #     else:
#       #         print("[WARNING] ⚠️ `pad_token_id` is missing! Fine-tuning may not work correctly.")

#       # Compute loss and backpropagation
      
#       # Forward pass
#       outputs = model(**inputs, labels=labels)
#       loss = outputs.loss
    
#       # Backward pass and optimization
#       loss.backward()
#       optimizer.step()
    
#       # Return stats for educational demonstration
#       stats = {
#           "status": "success",
#           "loss": float(loss.item()),
#           "text_length": len(text),
#           "unique_chars": len(set(text))
#       }
    
#       print(f"[SUCCESS] 🏆 Fine-tuning completed successfully! Loss: {loss.item():.4f}")
#       return stats

#     except Exception as e:
#         print(f"[WARNING] Fine-tuning error: {e}")
#         return {"status": "error", "error": str(e)}



