from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .india import LocationMode


class NominatimResolver:
    """Small policy-friendly OpenStreetMap reverse-geocoding client."""

    def __init__(self) -> None:
        self.enabled = os.getenv("NOMINATIM_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        self.base_url = os.getenv(
            "NOMINATIM_URL", "https://nominatim.openstreetmap.org/reverse"
        )
        self.user_agent = os.getenv(
            "NOMINATIM_USER_AGENT", "GeoManJi/0.1 academic-prototype"
        )
        self.timeout = float(os.getenv("NOMINATIM_TIMEOUT_SECONDS", "4"))
        self.delay = float(os.getenv("NOMINATIM_DELAY_SECONDS", "1.05"))
        self._cache: dict[tuple[float, float], dict[str, str] | None] = {}

    def resolve_many(
        self, modes: Sequence[LocationMode]
    ) -> list[dict[str, str] | None]:
        results: list[dict[str, str] | None] = []
        for index, mode in enumerate(modes):
            if index and self.enabled and self.delay > 0:
                time.sleep(self.delay)
            results.append(self.resolve(mode.latitude, mode.longitude))
        return results

    def resolve(self, latitude: float, longitude: float) -> dict[str, str] | None:
        if not self.enabled:
            return None
        key = (round(latitude, 4), round(longitude, 4))
        if key in self._cache:
            return self._cache[key]

        params = urlencode(
            {
                "format": "jsonv2",
                "lat": f"{latitude:.7f}",
                "lon": f"{longitude:.7f}",
                "zoom": 12,
                "addressdetails": 1,
                "accept-language": "en",
            }
        )
        request = Request(
            f"{self.base_url}?{params}",
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload: dict[str, Any] = json.load(response)
            address = payload.get("address") or {}
            if str(address.get("country_code", "")).lower() != "in":
                result = None
            else:
                locality = next(
                    (
                        address.get(field)
                        for field in (
                            "city",
                            "town",
                            "village",
                            "municipality",
                            "county",
                            "state_district",
                        )
                        if address.get(field)
                    ),
                    None,
                )
                state = address.get("state")
                name = locality or state or "India"
                short_display = ", ".join(
                    part for part in (locality, state, "India") if part
                )
                result = {
                    "name": str(name),
                    "display_name": short_display
                    or str(payload.get("display_name", name)),
                }
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
            result = None
        self._cache[key] = result
        return result
