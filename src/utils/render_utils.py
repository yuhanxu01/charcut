import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def load_font(font_path: str, size: int):
    return ImageFont.truetype(font_path, size)

def text_to_sprite(ch: str, font, size: int = 16) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(ch, font=font)
    draw.text(((size - w) / 2, (size - h) / 2), ch, font=font, fill=(0,0,0,255))
    return img

def paste_rgba(dst: Image.Image, src: Image.Image, xy):
    dst.alpha_composite(src, dest=xy)
    return dst

def random_affine(img: Image.Image, angle_deg: float, scale: float):
    w, h = img.size
    img2 = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.BILINEAR)
    img2 = img2.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
    return img2

def to_np(img: Image.Image):
    return np.array(img)

def add_canvas_noise(img: Image.Image, sigma=6.0):
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape[:2]).astype(np.int16)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr[:,:,:3] = np.clip(arr[:,:,:3] + noise[...,None], 0, 255)
    else:
        arr[:,:,:3] = np.clip(arr[:,:,:3] + noise[...,None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), img.mode)
