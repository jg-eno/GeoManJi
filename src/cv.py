"""Classical OpenCV stage: preprocessing, morphology, text-region and plate detection."""

import base64

import cv2
import numpy as np

# ponytail: Russian-plate Haar cascade ships with OpenCV; swap for a YOLO plate model if recall on Indian plates is too low
_PLATE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)
WORK_WIDTH = 800  # every photo is scaled to this width so kernel sizes mean the same thing
BLUR_THRESHOLD = 100.0
MIN_LETTERS = 5  # letter-like components a region needs to count as text
MAX_REGION_AREA = 0.15  # regions larger than this share of the image are background
MAX_CROPS = 4

TEXT_BOX = (0, 200, 0)
PLATE_BOX = (0, 0, 255)
REJECTED_BOX = (150, 150, 150)


def _jpeg(image, quality=82) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _draw(image, boxes, color, thickness=2):
    out = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def preprocess(image):
    """Resize to WORK_WIDTH, grayscale, CLAHE contrast equalisation, light Gaussian denoise."""
    scale = WORK_WIDTH / image.shape[1]
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return resized, gray, cv2.GaussianBlur(equalized, (3, 3), 0), scale


def sharpness(gray):
    """Variance of the Laplacian: low values mean a blurry image."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var()), cv2.convertScaleAbs(laplacian)


def letter_count(binary, box) -> int:
    """Connected components inside the box that look like letters of one text size."""
    x, y, w, h = box
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary[y : y + h, x : x + w])
    heights = np.array(
        [bh for _, _, bw, bh, area in stats[1:] if 8 <= bh <= h and 0.1 <= bw / bh <= 2.5 and area >= 20]
    )
    if len(heights) < 3:
        return 0
    median = np.median(heights)
    return int(np.sum(np.abs(heights - median) < 0.4 * median))


def text_regions(gray):
    """Black-hat/top-hat -> Otsu -> opening -> closing -> contours -> letter check -> NMS."""
    height, width = gray.shape
    stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 11))
    stages = {"binary": [], "opened": [], "closed": []}
    thresholds, candidates = [], []
    # Black-hat finds dark text on light signs, top-hat finds light text on dark.
    for name, op in (("blackhat", cv2.MORPH_BLACKHAT), ("tophat", cv2.MORPH_TOPHAT)):
        strokes = cv2.morphologyEx(gray, op, stroke_kernel)
        otsu, binary = cv2.threshold(strokes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, block_kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= 12 and 600 <= w * h <= MAX_REGION_AREA * height * width:
                candidates.append(([x, y, w, h], letter_count(opened, (x, y, w, h))))
        thresholds.append(round(otsu))
        stages[name] = strokes
        stages["binary"].append(binary)
        stages["opened"].append(opened)
        stages["closed"].append(closed)
    # Show both polarities in one mask per stage.
    for key in ("binary", "opened", "closed"):
        stages[key] = cv2.bitwise_or(*stages[key])

    accepted = [(box, n) for box, n in candidates if n >= MIN_LETTERS]
    keep = cv2.dnn.NMSBoxes([b for b, _ in accepted], [float(n) for _, n in accepted], 0, 0.3)
    # Most letters first, so the best regions are cropped for OCR.
    regions = sorted((accepted[i] for i in np.array(keep).flatten()), key=lambda r: -r[1])
    return [box for box, _ in regions], candidates, stages, thresholds


def plates(gray) -> list[list[int]]:
    found = _PLATE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [list(map(int, box)) for box in found]


def _crop(image, box, scale, pad=0.1):
    """Cut a box (in working coordinates) out of the full-resolution image."""
    x, y, w, h = (round(v / scale) for v in box)
    px, py = round(w * pad), round(h * pad)
    return image[max(0, y - py) : y + h + py, max(0, x - px) : x + w + px]


def analyze(image_path) -> tuple[dict, list[bytes]]:
    """Run the pipeline; returns the API payload and JPEG crops for OCR."""
    original = cv2.imread(str(image_path))
    if original is None:
        return {"error": "OpenCV could not read the image"}, []
    image, gray, enhanced, scale = preprocess(original)

    score, laplacian = sharpness(gray)
    text_boxes, candidates, text_stages, thresholds = text_regions(enhanced)
    plate_boxes = plates(enhanced)
    crops = [_jpeg(_crop(original, box, scale), 90) for box in plate_boxes + text_boxes[:MAX_CROPS]]

    verified = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2
    for (x, y, w, h), n in candidates:
        ok = n >= MIN_LETTERS
        cv2.rectangle(verified, (x, y), (x + w, y + h), TEXT_BOX if ok else REJECTED_BOX, 2 if ok else 1)
        if ok:
            cv2.putText(verified, f"{n} letters", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_BOX, 1, cv2.LINE_AA)
    annotated = _draw(_draw(image, text_boxes, TEXT_BOX), plate_boxes, PLATE_BOX, 3)

    stages = [
        ("Grayscale", f"Resized to {WORK_WIDTH}px wide (×{scale:.2f}) and converted to one channel.", gray, "preprocess"),
        ("CLAHE", "Contrast-limited adaptive histogram equalisation (8×8 tiles), then 3×3 Gaussian blur.", enhanced, "preprocess"),
        ("Laplacian", f"Edge response. Variance = {score:.1f}; below {BLUR_THRESHOLD:.0f} counts as blurry.", laplacian, "preprocess"),
        ("Black-hat", "Closing minus image: dark strokes thinner than 15×15 (dark text on light signs).", text_stages["blackhat"], "morphology"),
        ("Top-hat", "Image minus opening: light strokes thinner than 15×15 (light text on dark signs).", text_stages["tophat"], "morphology"),
        ("Otsu threshold", f"Automatic global threshold per polarity: {thresholds}.", text_stages["binary"], "morphology"),
        ("Opening", "Erode then dilate with 3×3: removes speckle noise.", text_stages["opened"], "morphology"),
        ("Closing", "Dilate then erode with 41×11: merges letters into words and text blocks.", text_stages["closed"], "morphology"),
        ("Letter check", f"{len(candidates)} contour regions; connected components count letter-sized blobs of similar height. ≥{MIN_LETTERS} letters kept (green), the rest rejected (grey).", verified, "detection"),
        ("Text regions", f"Non-maximum suppression merges overlaps: {len(text_boxes)} text regions.", _draw(image, text_boxes, TEXT_BOX), "detection"),
        ("Haar cascade", f"Viola–Jones plate detector, scale 1.1, 5 neighbours: {len(plate_boxes)} found.", _draw(image, plate_boxes, PLATE_BOX, 3), "detection"),
    ]

    payload = {
        "sharpness": round(score, 2),
        "blurry": score < BLUR_THRESHOLD,
        "text_regions": text_boxes,
        "plates": plate_boxes,
        "stages": [
            {"name": name, "group": group, "description": text, "image": _data_url(_jpeg(img))}
            for name, text, img, group in stages
        ],
        "ocr_crops": [_data_url(crop) for crop in crops],
        "annotated_image": _data_url(_jpeg(annotated)),
    }
    return payload, crops


if __name__ == "__main__":
    canvas = np.full((300, 800), 255, np.uint8)
    cv2.putText(canvas, "MG ROAD BENGALURU", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    boxes, *_ = text_regions(canvas)
    assert boxes, "expected a text region on a synthetic sign"
    assert sharpness(canvas)[0] > sharpness(cv2.GaussianBlur(canvas, (21, 21), 0))[0]
    payload, crops = analyze(__file__.replace("src/cv.py", "src/sample_data/images.png"))
    assert payload["plates"] and payload["text_regions"] and crops
    print("ok")
