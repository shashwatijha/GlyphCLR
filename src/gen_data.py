import os
from PIL import Image, ImageDraw, ImageFont

# 1. Configuration
FONT_DIR = "fonts"
DATA_DIR = "data/glyphs"
IMG_SIZE = 128  
PADDING = 20  # Space between glyph and image edge

# SCRIPTS = {
#     "egyptian": ["NotoSansEgyptianHieroglyphs-Regular.ttf", range(0x13000, 0x1342F)],
#     "cuneiform": ["NotoSansCuneiform-Regular.ttf", range(0x12000, 0x123FF)],
#     "linear_b": ["NotoSansLinearB-Regular.ttf", range(0x10000, 0x100FF)],
# }
SCRIPTS = {
    "linear_a": ["NotoSansLinearA-Regular.ttf", range(0x10600, 0x1077F)],
}

def generate_tight_glyph(char, font, img_size, padding):
    # 1. Get the mask from the font
    mask = font.getmask(char)
    if mask.size[0] == 0 or mask.size[1] == 0:
        return None
    
    # 2. Convert mask to a proper Image object
    # We create an image the exact size of the mask
    glyph_img = Image.new('L', mask.size, 255)
    mask_img = Image.frombytes('L', mask.size, bytes(mask))
    glyph_img.paste(0, (0, 0, mask.size[0], mask.size[1]), mask=mask_img)
    
    # 3. Find the bounding box of the ink
    bbox = glyph_img.getbbox() 
    if not bbox: 
        return None
    
    # 4. Crop to the ink
    glyph_crop = glyph_img.crop(bbox)
    
    # 5. Resize glyph to fit within the padded box
    max_size = img_size - (padding * 2)
    w, h = glyph_crop.size
    ratio = min(max_size/w, max_size/h)
    new_size = (int(w * ratio), int(h * ratio))
    glyph_crop = glyph_crop.resize(new_size, Image.Resampling.LANCZOS)
    
    # 6. Paste into centered 128x128 white canvas
    final_img = Image.new('L', (img_size, img_size), 255)
    cw, ch = glyph_crop.size
    offset = ((img_size - cw) // 2, (img_size - ch) // 2)
    final_img.paste(glyph_crop, offset)
    
    return final_img
def build_dataset():
    for script, (font_file, code_range) in SCRIPTS.items():
        print(f"Processing {script}...")
        save_path = os.path.join(DATA_DIR, script)
        os.makedirs(save_path, exist_ok=True)
        
        font_path = os.path.join(FONT_DIR, font_file)
        # Use a large font size to ensure high-res mask before resizing
        font = ImageFont.truetype(font_path, 120)
        
        count = 0
        for code in code_range:
            char = chr(code)
            img = generate_tight_glyph(char, font, IMG_SIZE, PADDING)
            
            if img:
                img.save(f"{save_path}/{script}_{count:04d}.png")
                count += 1
                
        print(f"Finished {script}: {count} images saved.")

if __name__ == "__main__":
    build_dataset()