# Testing the image model

## The image model's Pillow fixtures (#322)

`crates/server/tests/fixtures/pillow_resize/` holds Pillow 12.3.0's Lanczos
resizes that `media::resize_rgba` must equal byte for byte, and
`crates/server/tests/fixtures/images/alpha_96x64.webp` an RGBA WebP condition
image. Each is Pillow's output for a vision fixture in
`crates/artifact/tests/fixtures/vision/images/` (recipe checked against the
committed pixels on 2026-10-14):

```python
from PIL import Image  # pillow==12.3.0
src = "crates/artifact/tests/fixtures/vision/images/"
out = "crates/server/tests/fixtures/"
alpha = Image.open(src + "alpha.png").convert("RGBA")
alpha.resize((160, 96), Image.LANCZOS).save(out + "pillow_resize/alpha_160x96.png")
alpha.crop((100, 70, 200, 130)).resize((160, 96), Image.LANCZOS).save(out + "pillow_resize/alpha_crop_160x96.png")
Image.open(src + "on_grid.png").convert("RGBA").resize((224, 160), Image.LANCZOS).save(out + "pillow_resize/on_grid_224x160.png")
alpha.resize((96, 64), Image.LANCZOS).save(out + "images/alpha_96x64.webp", lossless=True)
```
