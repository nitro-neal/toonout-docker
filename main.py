"""
ToonOut Background Removal API

A FastAPI service that removes backgrounds from anime/cartoon images using the
ToonOut model (fine-tuned BiRefNet).

Endpoints:
- GET /ping - Health check
- POST /cutout_zip - Upload ZIP of images, get ZIP of cutouts back
"""

import io
import os
import sys
import time
import zipfile
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFile, ImageOps

import torch
from torchvision import transforms

# Add BiRefNet to path
sys.path.insert(0, "/app/BiRefNet")

from models.birefnet import BiRefNet
from utils import check_state_dict

# Configuration
API_KEY = os.environ.get("TOONOUT_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
WEIGHTS_PATH = "/app/weights/birefnet_finetuned_toonout.pth"
IMAGE_SIZE = (1024, 1024)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Global model instance (lazy loaded)
_model = None


def load_model():
    """Load the ToonOut model (lazy initialization)."""
    global _model
    if _model is not None:
        return _model

    print(f"[model] Loading ToonOut model on {DEVICE}...")
    start = time.monotonic()

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    # Initialize BiRefNet without pretrained backbone
    model = BiRefNet(bb_pretrained=False)

    # Load fine-tuned weights
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = check_state_dict(state)
    model.load_state_dict(state, strict=False)

    # Move to device and set eval mode
    model.eval().to(DEVICE)
    if DEVICE == "cuda":
        model = model.half()

    _model = model
    print(f"[model] Model loaded in {time.monotonic() - start:.2f}s")
    return _model


def cutout_rgba(img: Image.Image, threshold: Optional[float] = None) -> Image.Image:
    """
    Remove background from image and return RGBA with transparency.

    Args:
        img: Input PIL Image
        threshold: Optional threshold (0-1) for hard mask cutoff

    Returns:
        RGBA PIL Image with transparent background
    """
    model = load_model()

    # Ensure RGB and handle EXIF orientation
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    original_size = rgb.size

    # Preprocess
    x = preprocess(rgb).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Inference
    with torch.inference_mode():
        pred = model(x)[-1].sigmoid().float().cpu()[0, 0]

    # Convert prediction to mask
    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(original_size, resample=Image.BILINEAR).convert("L")

    # Apply threshold if specified
    if threshold is not None:
        mask = mask.point(lambda p: 255 if p >= threshold * 255 else 0, mode="L")

    # Create output with alpha channel
    out = rgb.copy()
    out.putalpha(mask)

    return out


# FastAPI app
app = FastAPI(
    title="ToonOut API",
    description="Background removal API for anime/cartoon images using ToonOut",
    version="1.0.0",
)


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if one is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API key")


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "ToonOut API",
        "version": "1.0.0",
        "endpoints": {
            "/ping": "Health check",
            "/cutout_zip": "POST a ZIP of images to remove backgrounds",
        }
    }


@app.get("/ping", dependencies=[Depends(verify_api_key)])
def ping():
    """Health check endpoint."""
    return {"status": "ok", "device": DEVICE}


@app.post("/cutout_zip", dependencies=[Depends(verify_api_key)])
async def cutout_zip(
    file: UploadFile = File(..., description="ZIP file containing images"),
    threshold: Optional[float] = None,
):
    """
    Process a ZIP of images and return a ZIP of cutouts.

    Supported formats: PNG, JPG, JPEG, WEBP, BMP

    Args:
        file: ZIP file upload
        threshold: Optional threshold (0-1) for hard mask edges

    Returns:
        ZIP file containing processed images with _cutout.png suffix
    """
    # Validate file extension
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload must be a .zip file")

    # Read uploaded file
    raw = await file.read()

    try:
        zin = zipfile.ZipFile(io.BytesIO(raw), "r")
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid ZIP file")

    # Find image files
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    members: List[str] = [
        n for n in zin.namelist()
        if not n.endswith("/") and os.path.splitext(n)[1].lower() in valid_exts
    ]

    if not members:
        raise HTTPException(400, "ZIP contains no supported images")

    # Verify ZIP integrity
    corrupt_member = zin.testzip()
    if corrupt_member:
        raise HTTPException(400, f"ZIP integrity check failed for: {corrupt_member}")

    print(f"[zip] Processing {len(members)} image(s) from {file.filename}")

    # Process images and create output ZIP
    out_buf = io.BytesIO()

    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in members:
            try:
                info = zin.getinfo(name)
                print(f"[image] Processing: {name} ({info.file_size} bytes)")
                start = time.monotonic()

                # Load image
                with zin.open(name) as fp:
                    im = Image.open(io.BytesIO(fp.read()))
                    im.load()

                print(f"[image] Loaded {name} ({im.width}x{im.height})")

                # Process with ToonOut
                rgba = cutout_rgba(im, threshold)

                # Save to output ZIP
                base = os.path.splitext(os.path.basename(name))[0]
                out_name = f"{base}_cutout.png"

                png_bytes = io.BytesIO()
                rgba.save(png_bytes, format="PNG", optimize=True)
                png_bytes.seek(0)
                zout.writestr(out_name, png_bytes.read())

                elapsed = time.monotonic() - start
                print(f"[image] Completed {name} -> {out_name} in {elapsed:.2f}s")

            except Exception as e:
                print(f"[image] Error processing {name}: {e}")
                # Write error file for failed images
                error_name = f"{os.path.basename(name)}.ERROR.txt"
                zout.writestr(error_name, str(e))

    out_buf.seek(0)
    output_size = out_buf.getbuffer().nbytes
    print(f"[zip] Completed processing, output size: {output_size} bytes")

    headers = {"Content-Disposition": 'attachment; filename="cutouts.zip"'}
    return StreamingResponse(out_buf, media_type="application/zip", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1337)
