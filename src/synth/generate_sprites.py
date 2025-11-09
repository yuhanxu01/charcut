import argparse, os, glob
from PIL import Image
from utils.render_utils import load_font, text_to_sprite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", required=True, help="text file with characters")
    ap.add_argument("--fonts", required=True, help="dir with font files")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--size", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.charset, "r", encoding="utf-8") as f:
        charset = list(f.read().strip())
    font_files = [os.path.join(args.fonts, x) for x in os.listdir(args.fonts) if x.lower().endswith((".ttf",".otf"))]
    assert font_files, "no fonts found"
    for ch in charset:
        ch_dir = os.path.join(args.out, f"U{ord(ch):05X}")
        os.makedirs(ch_dir, exist_ok=True)
        for font_path in font_files:
            try:
                font = load_font(font_path, int(args.size*0.9))
                spr = text_to_sprite(ch, font, size=args.size)
                base = os.path.splitext(os.path.basename(font_path))[0]
                spr.save(os.path.join(ch_dir, f"{base}.png"))
            except Exception as e:
                print("font failed", font_path, e)
    print("done")

if __name__ == "__main__":
    main()
