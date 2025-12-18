# ToonOut Background Removal API

A Docker-based API service for removing backgrounds from anime/cartoon images using the [ToonOut model](https://huggingface.co/joelseytre/toonout) (fine-tuned BiRefNet).

## Features

- Upload a ZIP of images, get a ZIP of cutouts with transparent backgrounds
- Optimized for anime/cartoon style images
- Supports PNG, JPG, JPEG, WEBP, BMP formats
- Optional API key authentication
- Runs on CPU or GPU (NVIDIA CUDA)

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- ~2GB disk space for the model weights
- (Optional) NVIDIA GPU with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for GPU acceleration

### Step 1: Clone or Download This Repository

```bash
cd /path/to/TOONOUT-CUSTOM-VPS
```

Make sure you have these files:
```
TOONOUT-CUSTOM-VPS/
├── main.py
├── requirements.txt
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
└── docker-compose.gpu.yml
```

### Step 2: Build and Run the Container

**For CPU (works on any machine):**
```bash
docker-compose up --build
```

**For GPU (requires NVIDIA GPU + nvidia-docker):**
```bash
docker-compose -f docker-compose.gpu.yml up --build
```

The first build will take 5-15 minutes as it:
1. Downloads the Python base image
2. Installs dependencies (~1.5GB)
3. Clones the BiRefNet repository
4. Downloads the ToonOut model weights (~885MB)

### Step 3: Verify It's Running

Once you see output like:
```
toonout-api  | INFO:     Uvicorn running on http://0.0.0.0:1337
```

Test the health endpoint:
```bash
curl http://localhost:1337/ping
```

Expected response:
```json
{"status":"ok","device":"cpu"}
```

(Or `"device":"cuda"` if using GPU)

---

## Usage Guide

### Preparing Your Images

#### Option A: Create a ZIP from a folder of images

```bash
# If you have images in a folder called "my_images"
zip -r images.zip my_images/

# Or zip specific files
zip images.zip image1.png image2.jpg image3.webp
```

#### Option B: Create a ZIP from individual files

```bash
zip images.zip *.png *.jpg
```

### Example: Creating a Test ZIP

```bash
# Create a test folder
mkdir test_images

# Download some sample anime images (or use your own)
curl -o test_images/sample1.png "https://placekitten.com/400/400"
curl -o test_images/sample2.jpg "https://placekitten.com/500/500"

# Create the ZIP
zip -r images.zip test_images/
```

### Sending Images to the API

#### Basic Usage

```bash
curl -X POST "http://localhost:1337/cutout_zip" \
  -F "file=@images.zip" \
  -o cutouts.zip
```

This will:
1. Upload `images.zip` to the server
2. Process each image (remove background)
3. Download `cutouts.zip` with the results

#### With Threshold (Hard Edge Cutoff)

The `threshold` parameter (0.0 to 1.0) creates sharper edges:

```bash
# Soft edges (default, no threshold)
curl -X POST "http://localhost:1337/cutout_zip" \
  -F "file=@images.zip" \
  -o cutouts.zip

# Medium threshold
curl -X POST "http://localhost:1337/cutout_zip?threshold=0.5" \
  -F "file=@images.zip" \
  -o cutouts.zip

# Hard edges (binary mask)
curl -X POST "http://localhost:1337/cutout_zip?threshold=0.9" \
  -F "file=@images.zip" \
  -o cutouts.zip
```

#### With API Key Authentication

If you set `TOONOUT_API_KEY` environment variable:

```bash
curl -X POST "http://localhost:1337/cutout_zip" \
  -H "X-API-Key: your-secret-key" \
  -F "file=@images.zip" \
  -o cutouts.zip
```

### Extracting Results

```bash
# Unzip the results
unzip cutouts.zip -d output/

# List the output files
ls output/
# sample1_cutout.png
# sample2_cutout.png
# ...
```

---

## API Reference

### `GET /`

Returns API information.

**Response:**
```json
{
  "name": "ToonOut API",
  "version": "1.0.0",
  "endpoints": {
    "/ping": "Health check",
    "/cutout_zip": "POST a ZIP of images to remove backgrounds"
  }
}
```

### `GET /ping`

Health check endpoint.

**Headers (optional):**
- `X-API-Key`: Your API key (if configured)

**Response:**
```json
{"status": "ok", "device": "cpu"}
```

### `POST /cutout_zip`

Process a ZIP of images and return cutouts.

**Headers (optional):**
- `X-API-Key`: Your API key (if configured)

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | none | Value 0.0-1.0 for hard mask cutoff |

**Body:**
- `file`: ZIP file (multipart/form-data)

**Supported Image Formats:**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- WebP (.webp)
- BMP (.bmp)

**Response:**
- ZIP file containing processed images with `_cutout.png` suffix
- Failed images will have a `.ERROR.txt` file with the error message

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOONOUT_API_KEY` | (none) | Optional API key for authentication |

### Setting API Key

**Option 1: In docker-compose.yml**
```yaml
environment:
  - TOONOUT_API_KEY=your-secret-key-here
```

**Option 2: Using .env file**

Create a `.env` file:
```
TOONOUT_API_KEY=your-secret-key-here
```

**Option 3: Command line**
```bash
TOONOUT_API_KEY=your-secret-key docker-compose up
```

---

## Deployment Options

### Local Development

```bash
docker-compose up --build
```

### Production (Detached Mode)

```bash
# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### On a VPS/Cloud Server

1. SSH into your server
2. Install Docker and Docker Compose
3. Copy project files to server
4. Run:
```bash
docker-compose up -d --build
```

5. (Optional) Set up a reverse proxy (nginx/caddy) for HTTPS

### Example: Deploy to Ubuntu VPS

```bash
# On your VPS
sudo apt update
sudo apt install docker.io docker-compose -y

# Clone/copy your project
mkdir -p /opt/toonout
cd /opt/toonout
# ... copy files here ...

# Start the service
docker-compose up -d --build

# Check status
docker-compose ps
docker-compose logs -f
```

---

## GPU Support

For faster processing with NVIDIA GPU:

### Prerequisites

1. NVIDIA GPU with CUDA support
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Install NVIDIA Container Toolkit (Ubuntu)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Run with GPU

```bash
docker-compose -f docker-compose.gpu.yml up --build
```

Verify GPU is detected:
```bash
curl http://localhost:1337/ping
# {"status":"ok","device":"cuda"}
```

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Out of memory

The model requires ~2-4GB RAM. If you're running out of memory:
- Close other applications
- Increase Docker memory limit (Docker Desktop > Settings > Resources)
- Process fewer images at once

### Slow processing

- CPU processing takes ~5-15 seconds per image
- GPU processing takes ~0.5-2 seconds per image
- Consider using the GPU version for batch processing

### curl: Failed to open/read local data

The ZIP file doesn't exist at the specified path:
```bash
# Check if file exists
ls -la images.zip

# Use absolute path
curl -X POST "http://localhost:1337/cutout_zip" \
  -F "file=@/full/path/to/images.zip" \
  -o cutouts.zip
```

### 401 Unauthorized

API key is required but not provided:
```bash
curl -X POST "http://localhost:1337/cutout_zip" \
  -H "X-API-Key: your-api-key" \
  -F "file=@images.zip" \
  -o cutouts.zip
```

---

## Complete Example Workflow

```bash
# 1. Start the server (first time will take a while to build)
docker-compose up -d --build

# 2. Wait for it to be ready
sleep 30
curl http://localhost:1337/ping

# 3. Create a folder with your anime images
mkdir my_anime_images
cp ~/Downloads/*.png my_anime_images/

# 4. Zip the images
zip -r input.zip my_anime_images/

# 5. Process them
curl -X POST "http://localhost:1337/cutout_zip" \
  -F "file=@input.zip" \
  -o output.zip

# 6. Extract results
unzip output.zip -d results/

# 7. View your transparent PNGs!
open results/  # macOS
# or: xdg-open results/  # Linux
```

---

## License

- ToonOut Model: MIT License
- BiRefNet: MIT License
- This API wrapper: MIT License

## Credits

- [ToonOut](https://huggingface.co/joelseytre/toonout) by Joel Seytre & Matteo Muratori
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by ZhengPeng7
