# GeoManJi

FastAPI image-geolocation prototype for India. A photo goes through a classical
OpenCV stage, Gemini OCR, GeoCLIP and
[PLONK](https://github.com/nicolas-dufour/plonk); Gemini then fuses every
stage's output into up to three final place predictions, shown on a map with
routing from your current location.

## Project structure

```text
main.py                  FastAPI routes: /, /maps, POST /image
src/pipeline.py          Orchestrator: runs the stages (in parallel) and builds the response
src/cv.py                OpenCV stage: preprocessing, morphology, text regions, plates, OCR crops
src/gemini.py            Gemini client: OCR on photo + crops, and the final fusion step
src/plonk_india/         PLONK sampling, India mask, clustering, Nominatim place names
src/web/map.html         Single-page UI (upload, pipeline view, map, routing)
src/sample_data/         Example photos
```

```text
GeoCLIP  ──────────────────┐
PLONK India ───────────────┤  run in parallel
OpenCV ──► Gemini OCR ─────┘
             │
evidence matching ──► Gemini fusion ──► final places
```

## Installation

```bash
uv sync
```

## Start the server

Choose either command:

```bash
uv run uvicorn main:app --reload
```

```bash
uv run fastapi dev main.py
```

The API runs at `http://127.0.0.1:8000`.

Open the interactive upload/map interface at:

```text
http://127.0.0.1:8000/maps
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

## Run inference

Send an image using the `img` form field:

```bash
curl -X POST http://127.0.0.1:8000/image \
  -F "img=@src/sample_data/img2.png"
```

The response has one key per stage:

| Key | Contents |
|---|---|
| `final` | Up to three ranked places from the Gemini fusion step, with confidence, reasoning and supporting sources |
| `geoclip` | GeoCLIP's coordinate and probability |
| `plonk` | Up to five India-only PLONK candidates (`places`) and sample counts |
| `ocr` | Text, languages, location clues and area guess read by Gemini |
| `cv` | OpenCV results, stage images and the crops sent to OCR |

Any stage that fails reports an `error` field; the other stages still run.

PLONK downloads the `nicolas-dufour/PLONK_OSV_5M` checkpoint on first use, so the
first request can take several minutes and needs network access. A CUDA GPU is
recommended; GeoCLIP and PLONK are loaded lazily and cached after loading.

## OCR language analysis

Put your key in `.env` before starting the service:

```bash
GEMINI_API_KEY="your-key"
```

The application loads `.env` automatically. Do not commit the key.

Gemini reads the text in the image and extracts languages and location clues.
If Gemini is unavailable, the OCR fields are empty and the reason is reported in
`analysis.error`; GeoCLIP, PLONK and the OpenCV pipeline still run.

## Classical CV pipeline

`src/cv/classic.py` runs OpenCV on every upload, before OCR:

1. Preprocessing: resize to 800px wide, grayscale, CLAHE, Gaussian blur, and a
   Laplacian-variance blur check.
2. Morphology: black-hat/top-hat (dark and light text), Otsu threshold, opening
   to remove noise, closing to merge letters into words and blocks.
3. Detection: contours, a connected-component letter check (regions need five
   or more letter-sized blobs of similar height), non-maximum suppression, and a
   Haar-cascade number-plate detector.
4. Hand-off: text regions and plates are cropped from the full-resolution photo
   and sent to Gemini with the photo, so small or distant text is easier to read.

Each stage image is returned under `cv.stages` and shown on `/maps`.

## Final prediction

After every other stage has run, Gemini receives the photo plus a summary of
the CV results, OCR text and clues, the GeoCLIP coordinate and the PLONK
candidates, and returns up to three ranked places under `final.places`. Each
place has coordinates, a confidence, a short reasoning and the stages that
support it. If Gemini is unavailable, `final.error` explains why and the other
outputs are still returned.

## PLONK India settings

The defaults are suitable for a GPU prototype. They can be adjusted with:

```bash
export PLONK_MODEL="nicolas-dufour/PLONK_OSV_5M"
export PLONK_SAMPLES=1024
export PLONK_NUM_STEPS=32
export PLONK_CFG=0.0
export PLONK_CLUSTER_RADIUS_KM=125
export NOMINATIM_USER_AGENT="GeoManJi/0.1 your-contact@example.com"
```

PLONK has no native country constraint. GeoManJi therefore retains only samples
inside an India polygon/island mask, clusters nearby samples, and returns the
five strongest modes. `sample_share` is the fraction of retained Indian samples
in a mode; it is not a calibrated probability. If fewer than five Indian modes
exist, the API returns the available matches and explains this in `plonk.error`.

Nominatim only supplies human-readable labels; predictions remain usable if it
is unavailable. Set `NOMINATIM_ENABLED=false` to skip reverse geocoding.

## Tests

```bash
uv run python -m src.cv
```

The current GeoCLIP inference takes approximately 2–3 minutes per query.

## Project status

The geolocation and India map prototype is approximately 75% complete. Travel
destination enrichment, recommendations, and multi-stop routing are the major
remaining work. See [`task75.md`](task75.md) for the implementation report and
the firm completion plan.
