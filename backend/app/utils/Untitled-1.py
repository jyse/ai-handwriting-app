
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

# Initialize FastAPI app
app = FastAPI(title="Handwriting to Font Converter")

# ==================== SETUP DIRECTORIES & STATIC FILES ====================

# Set up static files and templates (will create if they don't exist)
for directory in ["static", "templates", "uploads", "fonts"]:
    os.makedirs(directory, exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")

print("🐲[INFO] Directories created: uploads/, fonts/, templates/, static/")

# ==================== WEB ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    print("[INFO] 🌳 Root endpoint accessed")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Fallback HTML with improved UI
        return HTMLResponse(content="""
        <html>
            <head>
                <title>Handwriting to Font Converter</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1, h2 { color: #333; }
                    .container { max-width: 800px; margin: 0 auto; padding: 20px; }
                    .upload-form { background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .btn { 
                        background-color: #4CAF50; 
                        color: white; 
                        padding: 10px 15px; 
                        border: none; 
                        border-radius: 4px; 
                        cursor: pointer; 
                        text-decoration: none;
                        display: inline-block;
                        margin-top: 10px;
                    }
                    .btn:hover { background-color: #45a049; }
                    #result { display: none; margin-top: 20px; padding: 20px; background-color: #f0f0f0; border-radius: 8px; }
                    #preview { margin-top: 20px; font-size: 24px; }
                    .loader { 
                        border: 5px solid #f3f3f3;
                        border-top: 5px solid #3498db;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        animation: spin 2s linear infinite;
                        display: none;
                        margin: 20px auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Handwriting to Font Converter</h1>
                    
                    <div class="upload-form">
                        <h2>Step 1: Get the Template</h2>
                        <p>Download our <a href="/template">handwriting template</a> and fill it in with your handwriting.</p>
                        
                        <h2>Step 2: Upload Your Handwriting</h2>
                        <form id="uploadForm">
                            <input type="file" name="file" accept="image/*" required>
                            <button type="submit" class="btn">Create My Font</button>
                        </form>
                        <div class="loader" id="loader"></div>
                    </div>
                    
                    <div id="result">
                        <h2>Your Font is Ready!</h2>
                        <p>Your handwriting has been converted to a font.</p>
                        <a id="downloadLink" href="#" class="btn">Download Font</a>
                        <a id="debugLink" href="#" target="_blank" style="margin-left: 10px;">View Extracted Letters</a>
                        
                        <div id="preview">
                            <h3>Preview:</h3>
                            <p id="previewText">The quick brown fox jumps over the lazy dog.</p>
                        </div>
                    </div>
                </div>
                
                <script>
                    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                        e.preventDefault();
                        
                        // Show loader
                        document.getElementById('loader').style.display = 'block';
                        
                        // Hide previous result if any
                        document.getElementById('result').style.display = 'none';
                        
                        const formData = new FormData(e.target);
                        
                        try {
                            const response = await fetch('/upload', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const data = await response.json();
                            
                            // Hide loader
                            document.getElementById('loader').style.display = 'none';
                            
                            if (data.success) {
                                // Show result section
                                document.getElementById('result').style.display = 'block';
                                
                                // Set download link
                                const downloadLink = document.getElementById('downloadLink');
                                downloadLink.href = data.download_url;
                                downloadLink.download = data.font_url.split('/').pop();
                                
                                // Set debug link
                                const debugLink = document.getElementById('debugLink');
                                debugLink.href = data.debug_url;
                                debugLink.textContent = 'View Extracted Letters';
                                
                                // Apply the font to preview text
                                const previewText = document.getElementById('previewText');
                                
                                // Create a style element to load the font
                                const style = document.createElement('style');
                                style.textContent = `
                                    @font-face {
                                        font-family: 'YourHandwriting';
                                        src: url('${data.download_url}') format('truetype');
                                    }
                                    #previewText {
                                        font-family: 'YourHandwriting', sans-serif;
                                    }
                                `;
                                document.head.appendChild(style);
                            } else {
                                alert('Error: ' + (data.error || 'Unknown error'));
                            }
                        } catch (error) {
                            // Hide loader
                            document.getElementById('loader').style.display = 'none';
                            
                            console.error('Error:', error);
                            alert('Error: ' + error.message);
                        }
                    });
                </script>
            </body>
        </html>
        """)

@app.get("/template")
async def get_template():
    """Provide a template for handwriting"""
    template_path = "static/handwriting_template.png"
    
    # Create template if it doesn't exist
    if not os.path.exists(template_path):
        create_template(template_path)
    
    return FileResponse(template_path)

def cleanup_old_sessions(max_sessions=10):
    """Clean up old upload sessions to prevent accumulation of files"""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        return
        
    # List all session directories
    sessions = []
    for item in os.listdir(uploads_dir):
        item_path = os.path.join(uploads_dir, item)
        if os.path.isdir(item_path):
            # Get creation time
            created_time = os.path.getctime(item_path)
            sessions.append((item, created_time))
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only the most recent sessions
    if len(sessions) > max_sessions:
        for session, _ in sessions[max_sessions:]:
            session_path = os.path.join(uploads_dir, session)
            try:
                shutil.rmtree(session_path)
                print(f"[INFO] Cleaned up old session: {session}")
            except Exception as e:
                print(f"[WARNING] Failed to remove session {session}: {e}")

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Run cleanup before processing new upload
    cleanup_old_sessions()

    """Process uploaded handwriting image and convert to font"""
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
            "download_url": f"/download-font/{os.path.basename(font_result)}" if success else None,
            "session_id": session_id,
            "letters_dir": user_letters_dir, 
            "debug_url": f"/debug/{session_id}"
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
    
@app.get("/download-font/{font_name}")
async def download_font(font_name: str):
    """Direct font download endpoint"""
    font_path = os.path.join("fonts", font_name)
    
    if not os.path.exists(font_path):
        return JSONResponse(content={"error": "Font not found"}, status_code=404)
    
    return FileResponse(
        path=font_path, 
        filename=font_name,
        media_type="font/ttf"
    )


@app.get("/debug/{session_id}")
async def debug_letters(request: Request, session_id: str):
    """Debug endpoint to view extracted letters"""
    letters_dir = f"uploads/{session_id}/letters"
    
    if not os.path.exists(letters_dir):
        return HTMLResponse(content="Session not found or letters not extracted.")
    
    # Find all letter files
    file_infos = []
    for filename in sorted(os.listdir(letters_dir)):
        if filename.startswith("letter_") and (filename.endswith(".png") or filename.endswith(".svg")):
            try:
                code_part = filename.split("_")[1].split(".")[0]
                if code_part.isdigit():
                    code = int(code_part)
                    try:
                        char = chr(code)
                        file_infos.append({
                            "path": f"/uploads/{session_id}/letters/{filename}",
                            "code": code,
                            "char": char,
                            "filename": filename
                        })
                    except ValueError:
                        file_infos.append({
                            "path": f"/uploads/{session_id}/letters/{filename}", 
                            "code": code,
                            "char": "?",
                            "filename": filename
                        })
            except (IndexError, ValueError):
                pass
    
    # Sort by character code
    file_infos.sort(key=lambda x: x["code"])
    
    html_content = f"""
    <html>
        <head>
            <title>Letter Extraction Debug</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .letter-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }}
                .letter-cell {{ 
                    border: 1px solid #ccc; 
                    padding: 10px; 
                    display: flex; 
                    flex-direction: column; 
                    align-items: center;
                    background-color: #f9f9f9;
                }}
                .letter-image {{ 
                    width: 100px; 
                    height: 100px; 
                    background-color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .letter-info {{ text-align: center; margin-top: 5px; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Letter Extraction Debug</h1>
            <p>Session ID: {session_id}</p>
            <h2>Extracted Letters ({len(file_infos)})</h2>
            <div class="letter-grid">
    """
    
    for info in file_infos:
        html_content += f"""
                <div class="letter-cell">
                    <div class="letter-image">
                        <img src="{info['path']}" style="max-width: 100%; max-height: 100%;">
                    </div>
                    <div class="letter-info">
                        <strong>'{info['char']}'</strong> (Code: {info['code']})
                    </div>
                </div>
        """
    
    html_content += """
            </div>
        </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


# ==================== TEMPLATE GENERATION ====================

def create_template(path):
    """Create a template for handwriting with clear cell boundaries"""
    print("🐲🐲🐲🐲🐲")
    # Set the size of the template
    width, height = 1200, 1800
    margin = 50
    cell_size = 120
    
    # Create a white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Calculate grid dimensions
    cols = 6
    rows = 10  # Enough for a-z, A-Z
    
    # Draw the grid with labels
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Use a font for cell labels
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the grid
    char_index = 0
    for row in range(rows):
        for col in range(cols):
            if char_index >= len(chars):
                break
                
            # Calculate cell position
            x1 = margin + col * cell_size
            y1 = margin + row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Draw cell border (thicker for clarity)
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            
            # Draw character label in top-left corner
            label = chars[char_index]
            draw.text((x1 + 5, y1 + 5), label, fill='black', font=font)
            
            char_index += 1
    
    # Add instructions at the top
    instructions = "Instructions: Draw each letter inside its cell. Keep your handwriting within the boundaries."
    draw.text((margin, 20), instructions, fill='black', font=font)
    
    # Save the template
    img.save(path)
    print(f"[INFO] Created improved template at {path}")
    
    return path






# ==================== FINE TUNING FOR OCR MODEL ====================

