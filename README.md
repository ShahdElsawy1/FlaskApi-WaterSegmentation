# Water Segmentation from Multispectral Satellite Data

Binary water segmentation on 128 x 128 multispectral tiles, with two trained models and two ready-to-run Flask APIs (one per model). Upload a 12-channel tile and get a water mask back, from a REST endpoint or a simple web page.

> ## Download both projects
> **Google Drive folder: [[PASTE YOUR DRIVE LINK HERE](https://drive.google.com/drive/folders/1aT4fPNebg8OhHGTVOjYA57q4bV3DT-sD?usp=sharing)](PASTE_YOUR_DRIVE_LINK_HERE)**
>
> The Drive folder contains one sub-folder per model, each with the code, the exported model and its own README:
> `water-segmentation-api/` (Model A) and `water-segmentation-api-3ch/` (Model B).

## The two models

| | Model A: 16-channel | Model B: 3-channel |
|---|---|---|
| Notebook | `twosplitswsp.ipynb` | `water_3ch_final_recipe.ipynb` |
| Input to the network | 16 channels: 6 bands, 4 spectral indices, terrain and land-cover features | 3 channels chosen from the best candidates (see below) |
| First convolution | Rewritten for 16 channels (RGB filters copied, other channels zero-initialised) | Unchanged pretrained ImageNet convolution |
| Encoder | ResNet50 (ImageNet) | ResNet50 (ImageNet) |
| Recipe | BCE + Tversky loss, channel attention, EMA, 2x input upscaling | Same recipe, carried over without re-tuning |
| API folder | `water-segmentation-api/` (port 5000) | `water-segmentation-api-3ch/` (port 5001) |

Both APIs take the same input: the 12 raw bands. The features each model needs are built inside the API with the same code as in training.

### Model B: how the 3 channels are chosen

Model B follows the mentor's request to use only the best three channels, or averages of channels, so the pretrained first layer can be used as it is. Four inputs are trained with the final recipe and the best validation IoU wins:

- **RGB (control):** Red, Green, Blue.
- **Top-3 by AUC:** the three channels that best separate water from background on the training pixels, skipping channels that are highly correlated with one already chosen.
- **Green, NIR, SWIR1:** the bands behind the NDWI and MNDWI water indices.
- **Group averages:** mean of the visible bands, NIR, and mean of the two SWIR bands.

## Results

Validation set (61 tiles), same split and seed for every model. IoU is computed over all pixels.

| Model | Val IoU | Val IoU with TTA |
|---|---|---|
| Scratch U-Net (16 channels, baseline) | 0.7618 | n/a |
| **Model A:** pretrained ResNet50 U-Net, 16 channels | 0.8385 | 0.8512 |
| **Model B:** pretrained ResNet50 U-Net, 3 channels | _fill in_ | _fill in_ |

Model A over three seeds: val IoU 0.8374 ± 0.0043. Model B selected channels: _fill in from `selected_channels_3ch.json`_.

TTA = test-time augmentation: 8 flips and rotations, with the predicted probabilities averaged.

## Data and method

- **Data:** 306 tiles of 128 x 128 x 12 with binary water masks. Water is about 26% of pixels, and about 15% of tiles contain no water.
- **Bands (in order):** Coastal aerosol, Blue, Green, Red, NIR, SWIR1, SWIR2, QA band, MERIT DEM, Copernicus DEM, ESA WorldCover, Water occurrence.
- **Derived features:** NDWI, MNDWI, NDVI, AWEI, slope and relief (from the Copernicus DEM, 30 m pixels), and water / wetland flags from ESA WorldCover.
- **Preprocessing:** per-channel clipping at the 1st and 99th percentiles of the training set, then z-scoring.
- **Split:** 80 / 20 train / validation (245 / 61 tiles), fixed seed.
- **Training:** AdamW, learning rate 3e-4 for the encoder and 1e-3 for the rest, 3 warm-up epochs then cosine decay, 150 epochs, batch size 32, random flips and 90-degree rotations. The final-epoch weights are used.
- **Model:** U-Net from `segmentation-models-pytorch` with an ImageNet-pretrained ResNet50 encoder. The input is upscaled 2x and the logits are average-pooled back to 128 x 128, which helps small water bodies.

## Quick start

Each folder is self-contained. Python 3.10 to 3.12 is recommended. Replace `<folder>` with `water-segmentation-api` (Model A) or `water-segmentation-api-3ch` (Model B).

```bash
cd <folder>
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 for Model A or http://127.0.0.1:5001 for Model B. The exported model is already in each folder's `model/` directory in the Drive download. To rebuild it from the notebook outputs, see the folder's own README (`export_model.py`).

## API

Both APIs expose the same endpoints.

| Endpoint | Description |
|---|---|
| `GET /` | Web page: upload a tile, see the RGB preview, mask, overlay and water coverage |
| `GET /health` | Model settings and device |
| `POST /predict` | Multipart upload in the field `file` (12-channel `.tif`, `.tiff` or `.npy`). Returns a PNG mask (0 = background, 255 = water) |

`/predict` options: `format=json` returns the water fraction and base64 images (add `array=1` for the mask as a 0/1 list), `tta=1` turns on test-time augmentation, `threshold=0.5` sets the probability cut-off. Errors come back as JSON with HTTP 400, 413 or 500.

```bash
curl -X POST -F "file=@sample.tif" http://127.0.0.1:5000/predict --output mask.png
```

## Testing

With the server running, `python test_api.py` (in a second terminal) sends every tile in `samples/` to the API, checks the response, computes IoU against the ground-truth masks when available, and writes `TEST_RESULTS.md` and the masks in `results/`.

## Limitations

- The dataset is small (306 tiles), so differences of a few thousandths in IoU are within seed noise.
- The validation set was used to choose the recipe and the input channels, so the reported scores are slightly optimistic.
- The models were trained on 128 x 128 tiles. Other sizes are padded and cropped, and may lose some accuracy.
- Small water bodies (a few pixels) remain the hardest case.

## Repository layout

```
.
├── README.md
├── water-segmentation-api/        # Model A: Flask API for the 16-channel model
├── water-segmentation-api-3ch/    # Model B: Flask API for the 3-channel model
└── notebooks/
    ├── twosplitswsp.ipynb
    └── water_3ch_final_recipe.ipynb
```
