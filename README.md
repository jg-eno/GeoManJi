# GeoManJi

FastAPI service for image geolocation with GeoCLIP.

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

Health check:

```bash
curl http://127.0.0.1:8000/
```

## Run inference

Send an image using the `img` form field:

```bash
curl -X POST http://127.0.0.1:8000/geolocate \
  -F "img=@src/sample_data/img2.png"
```

The response contains the predicted latitude, longitude, probability, OCR evidence, detected languages, and any Gemini-derived area clues.

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

The current GeoCLIP inference takes approximately 2–3 minutes per query.

See [`docs/running-and-testing.md`](docs/running-and-testing.md) for complete setup, API checks, and inference testing instructions.

## Next step

Reduce GeoCLIP inference latency from 2–3 minutes per query.
