# Shared Test Assets

Reusable test files for model demos and verification.

| File | Type | Description |
|------|------|-------------|
| `test_image.jpg` | Image | General-purpose test image for SAM, image gen models |
| `test_speech.wav` | Audio | Short speech clip for Whisper |

## Usage

```powershell
# SAM interactive segmentation
python models/sam-3/model.py --image models/assets/test_image.jpg --interactive

# Whisper speech-to-text
python models/whisper/model.py --audio models/assets/test_speech.wav

# Image generation models write output here too
python models/sdxl-turbo/model.py --prompt "a cat" --output models/assets/output.png
```

## Adding Assets

Place shared test files here instead of per-model directories.
Keep files small (< 1 MB for images, < 5 MB for audio).
