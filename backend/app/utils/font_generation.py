import os
import subprocess
import uuid

def generate_font(text, letters_dir):
    """Generate a font using FontForge"""
    font_name = f"handwriting_{uuid.uuid4().hex[:8]}"
    script_path = f"generate_font_{font_name}.pe"
    font_path = os.path.join("fonts", f"{font_name}.ttf")

    # Make sure the fonts directory exists
    os.makedirs("fonts", exist_ok=True)

    # Ensure FontForge is installed
    try:
        subprocess.run(["fontforge", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARNING] FontForge not found. Skipping font generation.")
        return {"status": "no_fontforge", "letter_dir": letters_dir}

    # Count available letters
    svg_letters = [f for f in os.listdir(letters_dir) if f.endswith(".svg")]
    png_letters = [f for f in os.listdir(letters_dir) if f.endswith(".png")]

  # Get a unique list of letter codes from filename patterns (letter_XX.png or letter_XX.svg)
    letter_codes = set()
    for filename in os.listdir(letters_dir):
        if filename.startswith("letter_") and (filename.endswith(".png") or filename.endswith(".svg")):
            try:
                # Extract the number part from filenames like letter_97.png
                code_part = filename.split("_")[1].split(".")[0]
                if code_part.isdigit():  # Make sure it's a valid integer
                    letter_codes.add(int(code_part))
            except (IndexError, ValueError):
                continue
    
    print(f"[INFO] Found {len(svg_letters)} SVG and {len(png_letters)} PNG letters")
    print(f"[INFO] Found {len(letter_codes)} unique character codes") 

    # Generate FontForge script
    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")
        f.write(f"SetFontNames('{font_name}', 'Handwriting', 'Regular');\n")
        
        # Counter for mapped chars
        chars_mapped = 0

        # Map each detected character code to its corresponding letter file
        for code in letter_codes:
            # Check for SVG first (preferred)
            svg_file = os.path.join(letters_dir, f"letter_{code}.svg")
            png_file = os.path.join(letters_dir, f"letter_{code}.png")
            
            letter_file = None
            if os.path.exists(svg_file):
                letter_file = svg_file
            elif os.path.exists(png_file):
                letter_file = png_file
                
            if letter_file:
                f.write(f'Select({code});\n')
                f.write(f'Import("{letter_file}");\n')
                
                # Set reasonable bounds for the character
                f.write('RemoveOverlap();\n')
                f.write('Simplify();\n')
                f.write('CorrectDirection();\n')
                chars_mapped += 1
                try:
                    char = chr(code)
                    print(f"[INFO] Mapped character '{char}' (code {code}) to {letter_file}")
                except ValueError:
                    print(f"[INFO] Mapped character code {code} to {letter_file}")
        
        # If we don't have any character-named files, fall back to numbered files
        if chars_mapped == 0:
            print("[WARNING] No character files found. Falling back to numbered files.")
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
                    f.write('RemoveOverlap();\n')
                    f.write('Simplify();\n')
                    f.write('CorrectDirection();\n')
                    f.write('Center();\n')
                    chars_mapped += 1
                    print(f"[INFO] Mapped character '{char}' to {svg_file}")
                elif os.path.exists(png_file):
                    f.write(f'Select({char_code});\n')
                    f.write(f'Import("{png_file}");\n')
                    f.write('RemoveOverlap();\n')
                    f.write('Simplify();\n')
                    f.write('CorrectDirection();\n')
                    f.write('Center();\n')
                    chars_mapped += 1
                    print(f"[INFO] Mapped character '{char}' to {png_file}")

        # Generate if we have any characters mapped
        if chars_mapped > 0:            
            # Set some basic font metrics
            f.write('SetOS2Value("Weight", 400);\n')  # Regular weight
            f.write('SetOS2Value("Width", 5);\n')     # Medium width
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
            return None # Return None instead of a dictionary
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FontForge error: {e}")
        if e.stderr:
            print(f"Details: {e.stderr.decode()}")
        return None # Return None instead of a dictionary

