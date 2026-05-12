import os
import cv2
import argparse

def preprocess_all(input_root, output_root, size=(224, 224)):
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, filename)
                
                # FIX IS HERE: Use os.path.relpath, not os.relpath
                relative_path = os.path.relpath(root, input_root)
                target_dir = os.path.join(output_root, relative_path)
                
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                img = cv2.imread(img_path)
                if img is not None:
                    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(target_dir, filename), resized)
                    
    print(f"Preprocessing complete. Check: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/glyphs')
    parser.add_argument('--output_dir', default='data/processed')
    args = parser.parse_args()
    preprocess_all(args.input_dir, args.output_dir)