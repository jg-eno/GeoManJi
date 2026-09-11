from geoclip import GeoCLIP
from loguru import logger

class GeoTag:
    def __init__(self):
        self.model = GeoCLIP()

    def predict(self,img):
        top_pred_gps, top_pred_prob = self.model.predict(img, top_k=1)
        lat, lon = top_pred_gps[0]
        logger.info(f"Latitude : {lat} Logitude : {lon}")
        return {"Latitude":lat.item(), "Longitude":lon.item(), "Probability":top_pred_prob[0].item()}