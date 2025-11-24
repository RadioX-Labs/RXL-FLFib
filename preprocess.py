import os
import numpy as np
import cv2
from PIL import Image

def crop_to_largest_contour(img, margin=10):
    """Crop to largest non-black area (using largest contour), with optional margin."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + margin * 2, img.shape[1] - x)
    h = min(h + margin * 2, img.shape[0] - y)
    cropped = img[y:y+h, x:x+w]
    return cropped

def min_max_normalize_0_1(img):
    """Apply min-max normalization to a numpy array image (H,W,C), scale each channel to [0,1] (float32)."""
    img = img.astype(np.float32)
    for c in range(img.shape[2]):
        channel = img[..., c]
        min_val, max_val = channel.min(), channel.max()
        if max_val > min_val:
            img[..., c] = (channel - min_val) / (max_val - min_val)
        else:
            img[..., c] = 0
    return img

def crop_and_normalize_image_pil(pil_img, margin=10):
    """Crop black border, then normalize, return PIL image for saving and float numpy array for ML."""
    img = np.array(pil_img)
    if img.shape[-1] == 4:  # RGBA
        img = img[...,:3]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cropped = crop_to_largest_contour(img_bgr, margin)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    normalized = min_max_normalize_0_1(cropped_rgb)
    normalized_to_save = (normalized * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(normalized_to_save), normalized

def process_folders_recursively(base_input_dir, base_output_dir, margin=10):
    for root, dirs, files in os.walk(base_input_dir):
        rel_path = os.path.relpath(root, base_input_dir)
        cur_output_dir = os.path.join(base_output_dir, rel_path)
        if not os.path.exists(cur_output_dir):
            os.makedirs(cur_output_dir)
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')) or fname.startswith('._'):
                continue
            input_path = os.path.join(root, fname)
            output_path = os.path.join(cur_output_dir, fname)
            try:
                pil_img = Image.open(input_path).convert('RGB')
                cropped_norm_pil, cropped_norm_np = crop_and_normalize_image_pil(pil_img, margin)
                cropped_norm_pil.save(output_path)
                print(f"Saved: {output_path}")
                # Optionally save .npy for ML:
                # np.save(output_path.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'), cropped_norm_np)
            except Exception as e:
                print(f"Failed {input_path}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Crop black borders and min-max normalize ultrasound images in all folders.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder (can be nested)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder (will preserve structure)')
    parser.add_argument('--margin', type=int, default=0, help='Contour margin (default 10)')

    args = parser.parse_args()

    process_folders_recursively(
        base_input_dir=args.input_dir,
        base_output_dir=args.output_dir,
        margin=args.margin
    )
