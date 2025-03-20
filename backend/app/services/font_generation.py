import os
import subprocess
import uuid

def generate_font(recognized_chars, letters_dir):
    """Generate a font using FontForge from extracted handwriting letters."""
    import os
    import subprocess
    import uuid
    
    font_name = f"handwriting_{uuid.uuid4().hex[:8]}"
    script_path = f"generate_font_{font_name}.pe"
    font_path = os.path.join("fonts", f"{font_name}.ttf")

    # Ensure font directory exists
    os.makedirs("fonts", exist_ok=True)

    # Ensure FontForge is installed
    try:
        subprocess.run(["fontforge", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARNING] FontForge not found. Skipping font generation.")
        return None

    # Prioritize SVG files over PNG files, but use either if available
    letter_files_dict = {}
    for filename in os.listdir(letters_dir):
        if filename.startswith("letter_"):
            # Extract the letter index
            try:
                letter_idx = int(filename.split("_")[1].split(".")[0])
                file_ext = os.path.splitext(filename)[1].lower()
                
                # If we already have this letter and the current one is SVG, replace
                if letter_idx in letter_files_dict:
                    if file_ext == ".svg":  # SVG takes precedence
                        letter_files_dict[letter_idx] = filename
                else:
                    letter_files_dict[letter_idx] = filename
            except (ValueError, IndexError):
                continue
    
    # Sort by index to maintain character order
    sorted_letter_files = [letter_files_dict[idx] for idx in sorted(letter_files_dict.keys())]
    
    print(f"[INFO] Found {len(sorted_letter_files)} unique extracted letters in {letters_dir}")

    if not sorted_letter_files:
        print("[ERROR] No extracted letters found. Cannot generate font.")
        return None

    # Make sure we don't have more characters than letters
    usable_chars = recognized_chars[:min(len(sorted_letter_files), len(recognized_chars))]
    
    # Generate FontForge script
    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")
        f.write(f"SetFontNames('{font_name}', 'Handwriting', 'Your Handwriting');\n")
        f.write("Reencode('unicode');\n")  # Set encoding to Unicode
        f.write("SetFontOrder(2);\n")      # TrueType quadratic splines

        mapped_chars = 0

        # Map sorted files to recognized characters 
        for i, (char, letter_file) in enumerate(zip(usable_chars, sorted_letter_files)):
            letter_path = os.path.join(letters_dir, letter_file)
            
            # Skip if file doesn't exist
            if not os.path.exists(letter_path):
                print(f"[WARNING] File not found: {letter_path}")
                continue
                
            # Use ASCII value for character
            code = ord(char) if len(char) == 1 else (65 + i)  # Fallback to A-Z range if not a single char
            
            f.write(f'Select({code});\n')
            f.write(f'Import("{letter_path}");\n')
            f.write('RemoveOverlap();\n')
            f.write('Simplify();\n')
            f.write('CorrectDirection();\n')
            f.write('AutoWidth(20);\n')  # Add suitable spacing

            print(f"[INFO] Mapped '{char}' (code {code}) to {letter_path}")
            mapped_chars += 1

        # Ensure we have mapped characters
        if mapped_chars == 0:
            print("[ERROR] No letters mapped to characters. Font generation aborted.")
            os.remove(script_path)
            return None
        
        # Set font properties - SIMPLIFIED to be compatible with older FontForge versions
        f.write('SetOS2Value("Weight", 400);\n')
        f.write('SetOS2Value("Width", 5);\n')
        f.write(f'Generate("{font_path}");\n')
        f.write('Quit(0);\n')

    # Run FontForge script
    try:
        result = subprocess.run(["fontforge", "-script", script_path], check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            print(f"[INFO] FontForge output:\n{result.stdout.decode()}")

        if os.path.exists(script_path):
            os.remove(script_path)

        if os.path.exists(font_path):
            print(f"[SUCCESS] Font successfully generated: {font_path}")
            return font_path
        else:
            print("[ERROR] Font file was not created.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FontForge failed: {e}")
        if e.stderr:
            print(f"Details:\n{e.stderr.decode()}")
        if os.path.exists(script_path):
            os.remove(script_path)
        return None