"""OpenCV stage: preprocessing, morphology, DB text detection, text rectification and OCR crop enhancement."""

import base64
import threading
from functools import cache

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

# PP-OCRv3 DB text detector from the OpenCV model zoo (2.4 MB ONNX), fetched once into the Hugging Face cache.
DB_REPO, DB_FILE = "opencv/text_detection_ppocr", "text_detection_en_ppocrv3_2023may.onnx"
# ponytail: one lock around DB inference (cv2.dnn nets are not thread-safe); one net per thread if requests queue up
_db_lock = threading.Lock()

WORK_WIDTH = 800  # every photo is scaled to this width so kernel sizes mean the same thing
BLUR_THRESHOLD = 100.0
MIN_LETTERS = 4  # letter-like components on one text line a region needs to count as text
MAX_REGION_AREA = 0.15  # morphology regions larger than this share of the image are background
MAX_CROPS = 5
CROP_HEIGHT = 128  # shorter crops are upscaled before OCR

TEXT_BOX = (0, 200, 0)
DB_BOX = (255, 140, 0)
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


def _overlap(a, b) -> float:
    """Share of box a covered by box b."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    return max(0, w) * max(0, h) / (aw * ah)


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
    """Letter-like blobs of one text size that sit on a shared text line inside the box."""
    x, y, w, h = box
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary[y : y + h, x : x + w])
    blobs = np.array(
        [(by + bh / 2, bh) for _, by, bw, bh, area in stats[1:] if 8 <= bh <= h and 0.1 <= bw / bh <= 2.5 and area >= 20]
    )
    if len(blobs) < 3:
        return 0
    centres, heights = blobs.T
    median = np.median(heights)
    centres = centres[np.abs(heights - median) < 0.4 * median]
    # Letters of one line share a centre line; foliage and reflections scatter.
    return int(max(np.sum(np.abs(centres - c) < 0.5 * median) for c in centres)) if len(centres) else 0


def text_regions(gray):
    """Black-hat/top-hat -> Otsu -> opening -> closing -> contours -> letter check, at one scale."""
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
    return candidates, stages, thresholds


@cache
def _db_model():
    model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(hf_hub_download(DB_REPO, DB_FILE)))
    model.setBinaryThreshold(0.3)
    model.setPolygonThreshold(0.5)
    model.setUnclipRatio(2.0)
    model.setMaxCandidates(200)
    return model


def learned_regions(image) -> list[np.ndarray]:
    """Word and line quadrilaterals (4 corner points) from the DB (Differentiable Binarization) text detector."""
    height, width = image.shape[:2]
    with _db_lock:
        model = _db_model()
        # DB needs input sides that are multiples of 32; quads come back in image coordinates.
        model.setInputParams(1 / 255, (max(32, width // 32 * 32), max(32, height // 32 * 32)), (122.68, 116.67, 104.0))
        quads, _ = model.detect(image)
    return [np.array(quad, np.float32) for quad in quads]


def _corners(box) -> np.ndarray:
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def merge_regions(morph_boxes, quads, shape) -> list[tuple[list[int], np.ndarray, int]]:
    """Paint every region on a mask and close small gaps, so words of one sign become one block.
    Returns (block box, outline, member count); blocks the DB network confirms rank first.
    The outline is the member quad itself for a single DB line (keeps its perspective), otherwise
    the minimum-area rotated rectangle around all members (keeps their rotation)."""
    mask = np.zeros(shape[:2], np.uint8)
    for box in morph_boxes:
        cv2.fillPoly(mask, [_corners(box).astype(np.int32)], 255)
    for quad in quads:
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for contour in contours:
        block = list(cv2.boundingRect(contour))
        own_quads = [q for q in quads if _overlap(cv2.boundingRect(q), block) > 0.5]
        own_boxes = [_corners(b) for b in morph_boxes if _overlap(b, block) > 0.5]
        members = own_quads + own_boxes
        if not members:
            continue
        outline = own_quads[0] if len(own_quads) == 1 and not own_boxes else cv2.boxPoints(cv2.minAreaRect(np.vstack(members)))
        blocks.append((block, outline, len(own_quads), len(members)))
    blocks.sort(key=lambda b: (-b[2], -b[3], -b[0][2] * b[0][3]))
    return [(block, outline, n) for block, outline, _, n in blocks]


def order_corners(points) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left: smallest/largest x+y and y-x pick the corners."""
    points = np.asarray(points, np.float32)
    total, diff = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.array([points[total.argmin()], points[diff.argmin()], points[total.argmax()], points[diff.argmax()]])


def rectify(image, outline, pad=0.12) -> np.ndarray:
    """Perspective-warp a text outline to an upright, front-on rectangle (four-point transform)."""
    corners = order_corners(outline)
    centre = corners.mean(axis=0)
    corners = centre + (corners - centre) * (1 + pad)
    tl, tr, br, bl = corners
    width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, matrix, (max(1, width), max(1, height)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance(crop, blurry) -> np.ndarray:
    """Upscale small crops (bicubic), CLAHE on the L channel (contrast without shifting colours),
    then unsharp-mask, stronger when the photo is blurry."""
    if crop.shape[0] < CROP_HEIGHT:
        factor = min(4.0, CROP_HEIGHT / crop.shape[0])
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    amount = 1.0 if blurry else 0.5
    return cv2.addWeighted(crop, 1 + amount, cv2.GaussianBlur(crop, (0, 0), 2), -amount, 0)


def analyze(image_path) -> tuple[dict, list[bytes]]:
    """Run the pipeline; returns the API payload and JPEG crops for OCR."""
    original = cv2.imread(str(image_path))
    if original is None:
        return {"error": "OpenCV could not read the image"}, []
    image, gray, enhanced, scale = preprocess(original)

    score, laplacian = sharpness(gray)
    blurry = score < BLUR_THRESHOLD
    candidates, text_stages, thresholds = text_regions(enhanced)
    morph_boxes = [box for box, n in candidates if n >= MIN_LETTERS]
    quads = learned_regions(image)
    blocks = merge_regions(morph_boxes, quads, gray.shape)[:MAX_CROPS]
    text_boxes = [box for box, _, _ in blocks]
    # Outlines are in working coordinates; warp from the full-resolution photo so no detail is lost.
    crops = [_jpeg(enhance(rectify(original, outline / scale), blurry), 90) for _, outline, _ in blocks]

    verified = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2
    for (x, y, w, h), n in candidates:
        ok = n >= MIN_LETTERS
        cv2.rectangle(verified, (x, y), (x + w, y + h), TEXT_BOX if ok else REJECTED_BOX, 2 if ok else 1)
        if ok:
            cv2.putText(verified, f"{n} letters", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_BOX, 1, cv2.LINE_AA)
    outlines = [outline.astype(np.int32) for _, outline, _ in blocks]
    annotated = cv2.polylines(image.copy(), outlines, True, TEXT_BOX, 2)
    detected = cv2.polylines(_draw(image, morph_boxes, REJECTED_BOX, 1), [q.astype(np.int32) for q in quads], True, DB_BOX, 1)
    merged = cv2.polylines(detected.copy(), outlines, True, TEXT_BOX, 2)
    for (x, y, _, _), _, n in blocks:
        cv2.putText(merged, str(n), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_BOX, 1, cv2.LINE_AA)

    stages = [
        ("Grayscale", f"Resized to {WORK_WIDTH}px wide (×{scale:.2f}) and converted to one channel.", gray, "preprocess"),
        ("CLAHE", "Contrast-limited adaptive histogram equalisation (8×8 tiles), then 3×3 Gaussian blur.", enhanced, "preprocess"),
        ("Laplacian", f"Edge response. Variance = {score:.1f}; below {BLUR_THRESHOLD:.0f} counts as blurry{' (this one is: crops get stronger sharpening)' if blurry else ''}.", laplacian, "preprocess"),
        ("Black-hat", "Closing minus image: dark strokes thinner than 15×15 (dark text on light signs).", text_stages["blackhat"], "morphology"),
        ("Top-hat", "Image minus opening: light strokes thinner than 15×15 (light text on dark signs).", text_stages["tophat"], "morphology"),
        ("Otsu threshold", f"Automatic global threshold per polarity: {thresholds}.", text_stages["binary"], "morphology"),
        ("Opening", "Erode then dilate with 3×3: removes speckle noise.", text_stages["opened"], "morphology"),
        ("Closing", "Dilate then erode with 41×11: merges letters into words and text blocks.", text_stages["closed"], "morphology"),
        ("Letter check", f"{len(candidates)} contour regions; blobs of similar height on one text line are counted. ≥{MIN_LETTERS} letters kept (green), the rest rejected (grey).", verified, "detection"),
        ("DB text detector", f"PP-OCRv3 Differentiable Binarization network (OpenCV DNN): {len(quads)} word/line quadrilaterals (orange), any angle.", cv2.polylines(image.copy(), [q.astype(np.int32) for q in quads], True, DB_BOX, 2), "detection"),
        ("Merged regions", f"Morphology (grey) and DB (orange) regions painted on a mask and closed with 25×9. Each block is outlined by its DB quad or its minimum-area rotated rectangle: {len(text_boxes)} blocks (green, with member count).", merged, "detection"),
    ]

    payload = {
        "sharpness": round(score, 2),
        "blurry": blurry,
        "text_regions": text_boxes,
        "morphology_regions": len(morph_boxes),
        "db_regions": len(quads),
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
    candidates, *_ = text_regions(canvas)
    assert any(n >= MIN_LETTERS for _, n in candidates), "expected a text region on a synthetic sign"
    big = np.full((400, 800, 3), 255, np.uint8)
    cv2.putText(big, "VIETNAM", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 30)
    assert learned_regions(big), "DB should find bold text the 15x15 morphology kernel misses"
    assert sharpness(canvas)[0] > sharpness(cv2.GaussianBlur(canvas, (21, 21), 0))[0]
    # A sign rotated by 20° must come back upright: wider than tall after rectification.
    sign = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    tilted = cv2.warpAffine(sign, cv2.getRotationMatrix2D((400, 150), 20, 1), (800, 300), borderValue=(255, 255, 255))
    blocks = merge_regions([], learned_regions(tilted), tilted.shape)
    assert blocks, "DB should find the tilted sign"
    crop = rectify(tilted, blocks[0][1])
    assert crop.shape[1] > 3 * crop.shape[0], f"rectified crop should be a wide text line, got {crop.shape}"
    payload, crops = analyze(__file__.replace("src/cv.py", "src/sample_data/images.png"))
    assert payload["text_regions"] and crops
    print("ok")
