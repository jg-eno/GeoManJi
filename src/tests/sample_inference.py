from geoclip import GeoCLIP
from pathlib import Path


model = GeoCLIP()

img = "img2.png"
BASE_DIR = Path(__file__).resolve().parent
image_path = str(BASE_DIR / "src" / "sample_data" / img)


top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)

for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
    print(f"Probability: {top_pred_prob[i]:.6f}")
    print("")