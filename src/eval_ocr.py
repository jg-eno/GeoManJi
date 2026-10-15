"""Does the CV stage help OCR? Runs Gemini OCR on each labelled photo with and without the CV crops
and reports how many expected words each run read.

    uv run python -m src.eval_ocr
"""

import re
from pathlib import Path

from . import cv
from .gemini import Gemini

SAMPLES = Path(__file__).parent / "sample_data"
# ponytail: labels inline; move to a JSON file next to the photos once there are more than ~20
LABELS = {
    "hepburn.png": ["SASKATCHEWAN", "POOL", "ELEVATORS", "883"],
    "images.png": ["TN09BY9726", "MARUTI", "SUZUKI", "ZXI"],
    "location.png": ["VIETNAM"],
    "screen.png": ["COLORS", "NATURE", "COLLECTION", "SCENERIES", "WORLD"],
}


def squash(text: str) -> str:
    """Upper-case letters and digits only, so 'TN 09 BY-9726' matches 'TN09BY9726'."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def main() -> None:
    gemini = Gemini()
    totals = {"photo": 0, "photo+crops": 0}
    expected = 0
    print(f"{'image':14} {'crops':>5} {'photo':>8} {'+crops':>8}  missed with crops")
    for name, words in LABELS.items():
        path = SAMPLES / name
        _, crops = cv.analyze(path)
        hits = {}
        for run, parts in (("photo", []), ("photo+crops", crops)):
            text = squash(gemini.read_text(path, parts).text)
            hits[run] = [w for w in words if w in text]
            totals[run] += len(hits[run])
        expected += len(words)
        missed = [w for w in words if w not in hits["photo+crops"]]
        print(f"{name:14} {len(crops):>5} {len(hits['photo'])}/{len(words):<6} {len(hits['photo+crops'])}/{len(words):<6}  {' '.join(missed) or '-'}")
    print(f"{'total':14} {'':>5} {totals['photo']}/{expected:<6} {totals['photo+crops']}/{expected:<6}")


if __name__ == "__main__":
    main()
