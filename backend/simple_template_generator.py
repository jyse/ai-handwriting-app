from PIL import Image, ImageDraw, ImageFont

# Create a white background
width, height = 1200, 800
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Setup directory
import os
os.makedirs("static", exist_ok=True)

# Define the grid
margin = 50
grid_width = width - (2 * margin)
grid_height = height - (2 * margin)

# Grid for a-z and A-Z
cols = 6
rows = 9

# Calculate box size
box_width = grid_width // cols
box_height = grid_height // rows

# Use default font
font = ImageFont.load_default()

# Draw the grid with labels
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_index = 0

for row in range(rows):
    for col in range(cols):
        if char_index < len(chars):
            # Calculate box position
            x1 = margin + (col * box_width)
            y1 = margin + (row * box_height)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # Draw the box
            draw.rectangle((x1, y1, x2, y2), outline='black', width=2)
            
            # Add the character label
            char = chars[char_index]
            draw.text((x1 + 10, y1 + 10), char, fill='black')
            
            char_index += 1

# Save the template
template_path = "static/handwriting_template.png"
image.save(template_path)
print(f"Template created at {template_path}")