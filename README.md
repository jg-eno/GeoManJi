# GeoManJi

FastAPI image-geolocation prototype combining GeoCLIP, Gemini OCR, and
[PLONK](https://github.com/nicolas-dufour/plonk). PLONK's stochastic predictions
are filtered to India, grouped into five location modes, named through
OpenStreetMap Nominatim, and displayed as numbered map pins.

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

`/map` remains available as an alias.

Health check:

```bash
curl http://127.0.0.1:8000/
```

## Run inference

Send an image using the `img` form field. `/image` is the primary upload route;
`/geolocate` remains available as a backward-compatible alias:

```bash
curl -X POST http://127.0.0.1:8000/image \
  -F "img=@src/sample_data/img2.png"
```

The response contains the GeoCLIP coordinate, OCR evidence, detected languages,
Gemini-derived area clues, and up to five India-only PLONK matches in
`similar_places`. The same matches are repeated under `map.locations` for map
clients.

PLONK downloads the `nicolas-dufour/PLONK_OSV_5M` checkpoint on first use, so the
first request can take several minutes and needs network access. A CUDA GPU is
recommended; GeoCLIP and PLONK are loaded lazily and cached after loading.

## OCR language analysis

Put your key in `.env` before starting the service:

```bash
GEMINI_API_KEY="your-key"
```

The application loads `.env` automatically. Do not commit the key.

The OCR engine defaults to English and Kannada. Set `OCR_LANGUAGES` to a comma-separated list of EasyOCR language codes for the signs you expect:

```bash
export OCR_LANGUAGES="en,hi,kn,ta,te,ml,ar"
```

If Gemini is unavailable or no text is recognized, the pipeline falls back to Unicode script detection. Gemini failures are reported in `analysis.error` without preventing OCR or GeoCLIP inference.

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
uv run python -m unittest discover -s tests -v
```

The current GeoCLIP inference takes approximately 2–3 minutes per query.

## Project status

The geolocation and India map prototype is approximately 75% complete. Travel
destination enrichment, recommendations, and multi-stop routing are the major
remaining work. See [`task75.md`](task75.md) for the implementation report and
the firm completion plan.
