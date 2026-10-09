import tempfile
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from src.geoclip.geotag import GeoTag

app = FastAPI()
_geotag: GeoTag | None = None
_geotag_lock = threading.Lock()


def get_geotag() -> GeoTag:
    global _geotag
    if _geotag is None:
        with _geotag_lock:
            if _geotag is None:
                _geotag = GeoTag()
    return _geotag


@app.get("/")
def health():
    return {"Response": "OK", "map": "/maps", "docs": "/docs"}


@app.get("/maps", response_class=HTMLResponse)
@app.get("/map", response_class=HTMLResponse)
def map_view():
    return (Path(__file__).parent / "src" / "web" / "map.html").read_text(
        encoding="utf-8"
    )


@app.post("/image")
@app.post("/geolocate")
async def geolocate(img: Annotated[UploadFile, File()]):
    image_bytes = await img.read()
    suffix = Path(img.filename or "image.png").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
        temp.write(image_bytes)
        temp.flush()
        return await run_in_threadpool(get_geotag().predict_with_evidence, temp.name)
