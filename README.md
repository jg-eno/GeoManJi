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

The response contains the predicted latitude, longitude, and probability:

```json
{
  "Latitude": 0.0,
  "Longitude": 0.0,
  "Probability": 0.0
}
```

The current GeoCLIP inference takes approximately 2–3 minutes per query.

## Next step

Reduce GeoCLIP inference latency from 2–3 minutes per query.
