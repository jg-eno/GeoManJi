import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from src.geoclip.geotag import GeoTag


app = FastAPI()
gt = GeoTag()


@app.get("/")
def health():
    return {"Response": "OK"}


@app.post("/geolocate")
async def geolocate(img: UploadFile = File(...)):
    image_bytes = await img.read()
    suffix = Path(img.filename or "image.png").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
        temp.write(image_bytes)
        temp.flush()
        return gt.predict_with_evidence(temp.name)
