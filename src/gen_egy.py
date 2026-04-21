from PIL import ImageFont

# Path to one of your downloaded fonts
font_path = "fonts/NotoSansEgyptianHieroglyphs-Regular.ttf"

try:
    font = ImageFont.truetype(font_path, 32)
    print("Font loaded successfully!")
except OSError:
    print("Could not find or load the font. Check your file path!")