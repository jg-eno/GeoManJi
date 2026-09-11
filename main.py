from fastapi import FastAPI, UploadFile, File
from src.geoclip.geotag import GeoTag
import tempfile

app = FastAPI()
gt = GeoTag()

@app.get("/")
def health():
    return {"Response": "OK"}

@app.post("/geolocate")
async def geolocate(img: UploadFile = File(...)):
    image_bytes = await img.read()
    with tempfile.NamedTemporaryFile(suffix=img.filename, delete=True) as temp:
        temp.write(image_bytes)
        temp.flush()
        return gt.predict(temp.name)