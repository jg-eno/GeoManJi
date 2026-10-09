import tempfile
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from src.pipeline import GeoPipeline

app = FastAPI(title="GeoManJi")
PAGE = Path(__file__).parent / "src" / "web" / "map.html"

_pipeline: GeoPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> GeoPipeline:
    """Load the models once, on the first request."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = GeoPipeline()
    return _pipeline


@app.get("/")
def health():
    return {"Response": "OK", "map": "/maps", "docs": "/docs"}


@app.get("/maps", response_class=HTMLResponse)
def map_view():
    return PAGE.read_text(encoding="utf-8")


@app.post("/image")
async def analyze_image(img: Annotated[UploadFile, File()]):
    suffix = Path(img.filename or "image.png").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(await img.read())
        temp.flush()
        return await run_in_threadpool(lambda: get_pipeline().run(temp.name))
