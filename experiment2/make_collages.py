"""One-time script: collage VRP / VRP+Term Slope / VRP+VVIX MA5 side-by-side for each strategy."""
from pathlib import Path
from PIL import Image

BASE = Path(__file__).parent / "output" / "expanding_window"
OUT = BASE / "collage"
OUT.mkdir(exist_ok=True)

STRATEGIES = [
    "symmetric",
    "asymmetric",
    "base_return_shift",
    "leveraged_symmetric",
    "leveraged_asymmetric",
    "leveraged_base_return_shift",
]

def image_path(folder: str, strategy: str, suffix: str) -> Path:
    filename = f"{strategy}_{suffix}.png"
    return BASE / folder / filename

def make_collage(strategy: str):
    paths = [
        image_path("VRP", strategy, "VRP"),
        image_path("VRP + Term Slope", strategy, "VRP_+_Term_Slope"),
        image_path("VRP + VVIX MA5", strategy, "VRP_+_VVIX_MA5"),
    ]

    imgs = [Image.open(p) for p in paths]

    # Resize all to the same height (use the minimum height)
    target_h = min(img.height for img in imgs)
    resized = []
    for img in imgs:
        if img.height != target_h:
            scale = target_h / img.height
            new_w = int(img.width * scale)
            img = img.resize((new_w, target_h), Image.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized)
    collage = Image.new("RGB", (total_w, target_h), color=(255, 255, 255))

    x = 0
    for img in resized:
        collage.paste(img, (x, 0))
        x += img.width

    out_path = OUT / f"{strategy}_collage.png"
    collage.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}")

for strategy in STRATEGIES:
    make_collage(strategy)

print("Done.")
