import os
import subprocess
import uuid

def generate_font(recognized_chars, letters_dir):
    """Generate a font using FontForge from extracted handwriting letters."""
    
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

    # Find extracted letter files
    letter_files = [f for f in os.listdir(letters_dir) if f.startswith("letter_") and (f.endswith(".svg") or f.endswith(".png"))]
    print(f"[INFO] Found {len(letter_files)} extracted letters in {letters_dir}")

    if not letter_files:
        print("[ERROR] No extracted letters found. Cannot generate font.")
        return None

    # Sort files by index to ensure correct order
    sorted_files = sorted(letter_files, key=lambda f: int(f.split("_")[1].split(".")[0]))

    # Generate FontForge script
    with open(script_path, "w") as f:
        f.write("#!/usr/bin/env fontforge\n")
        f.write("New();\n")
        f.write(f"SetFontNames('{font_name}', 'Handwriting', 'Regular');\n")

        mapped_chars = 0

        # Map sorted files to recognized characters (they should be in same order)
        for i, (char, letter_file) in enumerate(zip(recognized_chars, sorted_files)):
            letter_path = os.path.join(letters_dir, letter_file)
            
            # Use ASCII value for character
            code = ord(char) if len(char) == 1 else (65 + i)  # Fallback to A-Z range if not a single char
            
            f.write(f'Select({code});\n')
            f.write(f'Import("{letter_path}");\n')
            f.write('RemoveOverlap();\n')
            f.write('Simplify();\n')
            f.write('CorrectDirection();\n')

            print(f"[INFO] Mapped '{char}' (code {code}) to {letter_path}")
            mapped_chars += 1

        # Ensure we have mapped characters
        if mapped_chars == 0:
            print("[ERROR] No letters mapped to characters. Font generation aborted.")
            return None
        
        # Set font properties
        f.write('SetOS2Value("Weight", 400);\n')
        f.write('SetOS2Value("Width", 5);\n')
        f.write(f'Generate("{font_path}");\n')
        f.write('Quit(0);\n')

    # Run FontForge script
    try:
        result = subprocess.run(["fontforge", "-script", script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            print(f"[INFO] FontForge output:\n{result.stdout.decode()}")

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
        return None