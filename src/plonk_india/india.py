from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# A lightweight Natural Earth-derived mainland boundary. The two small boxes below
# retain Plonk samples for India's Lakshadweep and Andaman/Nicobar island groups.
# This is an inference filter, not a statement about administrative boundaries.
INDIA_MAINLAND: tuple[tuple[float, float], ...] = (
    (77.837451, 35.494010),
    (78.912269, 34.321936),
    (78.811086, 33.506198),
    (79.208892, 32.994395),
    (79.176129, 32.483780),
    (78.458446, 32.618164),
    (78.738894, 31.515906),
    (79.721367, 30.882715),
    (81.111256, 30.183481),
    (80.476721, 29.729865),
    (80.088425, 28.794470),
    (81.057203, 28.416095),
    (81.999987, 27.925479),
    (83.304249, 27.364506),
    (84.675018, 27.234901),
    (85.251779, 26.726198),
    (86.024393, 26.630985),
    (87.227472, 26.397898),
    (88.060238, 26.414615),
    (88.174804, 26.810405),
    (88.043133, 27.445819),
    (88.120441, 27.876542),
    (88.730326, 28.086865),
    (88.814248, 27.299316),
    (88.835643, 27.098966),
    (89.744528, 26.719403),
    (90.373275, 26.875724),
    (91.217513, 26.808648),
    (92.033484, 26.838310),
    (92.103712, 27.452614),
    (91.696657, 27.771742),
    (92.503119, 27.896876),
    (93.413348, 28.640629),
    (94.565990, 29.277438),
    (95.404802, 29.031717),
    (96.117679, 29.452802),
    (96.586591, 28.830980),
    (96.248833, 28.411031),
    (97.327114, 28.261583),
    (97.402561, 27.882536),
    (97.051989, 27.699059),
    (97.133999, 27.083774),
    (96.419366, 27.264589),
    (95.124768, 26.573572),
    (95.155153, 26.001307),
    (94.603249, 25.162495),
    (94.552658, 24.675238),
    (94.106742, 23.850741),
    (93.325188, 24.078556),
    (93.286327, 23.043658),
    (93.060294, 22.703111),
    (93.166128, 22.278460),
    (92.672721, 22.041239),
    (92.146035, 23.627499),
    (91.869928, 23.624346),
    (91.706475, 22.985264),
    (91.158963, 23.503527),
    (91.467730, 24.072639),
    (91.915093, 24.130414),
    (92.376202, 24.976693),
    (91.799596, 25.147432),
    (90.872211, 25.132601),
    (89.920693, 25.269750),
    (89.832481, 25.965082),
    (89.355094, 26.014407),
    (88.563049, 26.446526),
    (88.209789, 25.768066),
    (88.931554, 25.238692),
    (88.306373, 24.866079),
    (88.084422, 24.501657),
    (88.699940, 24.233715),
    (88.529770, 23.631142),
    (88.876312, 22.879146),
    (89.031961, 22.055708),
    (88.888766, 21.690588),
    (88.208497, 21.703172),
    (86.975704, 21.495562),
    (87.033169, 20.743308),
    (86.499351, 20.151638),
    (85.060266, 19.478579),
    (83.941006, 18.302010),
    (83.189217, 17.671221),
    (82.192792, 17.016636),
    (82.191242, 16.556664),
    (81.692719, 16.310219),
    (80.791999, 15.951972),
    (80.324896, 15.899185),
    (80.025069, 15.136415),
    (80.233274, 13.835771),
    (80.286294, 13.006261),
    (79.862547, 12.056215),
    (79.857999, 10.357275),
    (79.340512, 10.308854),
    (78.885345, 9.546136),
    (79.189720, 9.216544),
    (78.277941, 8.933047),
    (77.941165, 8.252959),
    (77.539898, 7.965535),
    (76.592979, 8.899276),
    (76.130061, 10.299630),
    (75.746467, 11.308251),
    (75.396101, 11.781245),
    (74.864816, 12.741936),
    (74.616717, 13.992583),
    (74.443859, 14.617222),
    (73.534199, 15.990652),
    (73.119909, 17.928570),
    (72.820909, 19.208234),
    (72.824475, 20.419503),
    (72.630533, 21.356009),
    (71.175273, 20.757441),
    (70.470459, 20.877331),
    (69.164130, 22.089298),
    (69.644928, 22.450775),
    (69.349597, 22.843180),
    (68.176645, 23.691965),
    (68.842599, 24.359134),
    (71.043240, 24.356524),
    (70.844699, 25.215102),
    (70.282873, 25.722229),
    (70.168927, 26.491872),
    (69.514393, 26.940966),
    (70.616496, 27.989196),
    (71.777666, 27.913180),
    (72.823752, 28.961592),
    (73.450638, 29.976413),
    (74.421380, 30.979815),
    (74.405929, 31.692639),
    (75.258642, 32.271105),
    (74.451559, 32.764900),
    (74.104294, 33.441473),
    (73.749948, 34.317699),
    (74.240203, 34.748887),
    (75.757061, 34.504923),
    (76.871722, 34.653544),
    (77.837451, 35.494010),
)


@dataclass(frozen=True)
class LocationMode:
    latitude: float
    longitude: float
    supporting_samples: int
    sample_share: float


def _point_in_polygon(
    latitude: float,
    longitude: float,
    polygon: Sequence[tuple[float, float]],
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > latitude) != (y2 > latitude)
        if crosses:
            intersection = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < intersection:
                inside = not inside
        previous = current
    return inside


def is_in_india(latitude: float, longitude: float) -> bool:
    """Return whether a coordinate is inside the prototype's India mask."""
    if _point_in_polygon(latitude, longitude, INDIA_MAINLAND):
        return True
    in_lakshadweep = 8.0 <= latitude <= 12.8 and 71.0 <= longitude <= 74.0
    in_andaman_nicobar = 6.5 <= latitude <= 14.0 and 92.0 <= longitude <= 94.5
    return in_lakshadweep or in_andaman_nicobar


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    value = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def india_only(
    coordinates: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [
        (float(latitude), float(longitude))
        for latitude, longitude in coordinates
        if math.isfinite(float(latitude))
        and math.isfinite(float(longitude))
        and is_in_india(float(latitude), float(longitude))
    ]


def rank_location_modes(
    coordinates: Sequence[tuple[float, float]],
    *,
    top_k: int = 5,
    cluster_radius_km: float = 125.0,
) -> list[LocationMode]:
    """Collapse stochastic PLONK samples into ranked, geographically distinct modes."""
    if top_k <= 0 or not coordinates:
        return []

    count = len(coordinates)
    neighbors = [{index} for index in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            if (
                haversine_km(coordinates[first], coordinates[second])
                <= cluster_radius_km
            ):
                neighbors[first].add(second)
                neighbors[second].add(first)

    remaining = set(range(count))
    clusters: list[list[int]] = []
    while remaining:
        seed = max(
            remaining,
            key=lambda index: sum(other in remaining for other in neighbors[index]),
        )
        members = sorted(remaining.intersection(neighbors[seed]))
        clusters.append(members)
        remaining.difference_update(members)

    clusters.sort(key=lambda members: (-len(members), min(members)))
    total = len(coordinates)
    modes: list[LocationMode] = []
    for members in clusters[:top_k]:
        # Choose an observed sample (the medoid), so an averaged coordinate cannot
        # drift across a coastline or national boundary.
        medoid = min(
            members,
            key=lambda candidate: sum(
                haversine_km(coordinates[candidate], coordinates[other])
                for other in members
            ),
        )
        latitude, longitude = coordinates[medoid]
        modes.append(
            LocationMode(
                latitude=latitude,
                longitude=longitude,
                supporting_samples=len(members),
                sample_share=len(members) / total,
            )
        )
    return modes
