"""Image importer — BW plan image → FloorPlan via OpenCV contours."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from floorplan.geometry.ops import aspect_ratio, bounding_box, to_points
from floorplan.schemas.enums import RoomType, ZoneType
from floorplan.schemas.models import Boundary, FloorPlan, Room
from floorplan.topology import build_adjacency


def _order_rooms_by_area(contours_data: list[dict]) -> list[dict]:
    return sorted(contours_data, key=lambda c: c["area"], reverse=True)


def _guess_room_type(area_sqft: float, ar: float, touching_exterior: bool, rank: int) -> RoomType:
    if area_sqft < 2.5:
        return RoomType.CLOSET
    if area_sqft < 4.5 and ar < 2.2:
        return RoomType.BATHROOM
    if area_sqft < 6 and ar > 2.5:
        return RoomType.HALLWAY
    if rank == 0 and touching_exterior:
        return RoomType.LIVING
    if area_sqft >= 8 and touching_exterior:
        return RoomType.BEDROOM
    if 4 <= area_sqft < 10:
        return RoomType.KITCHEN
    return RoomType.OTHER


def import_bw_image(
    image_bytes: bytes,
    *,
    feet_per_pixel: float = 0.05,
    min_area_sqft: float = 1.0,
    simplify_epsilon_px: float = 2.0,
) -> FloorPlan:
    """Pipeline: grayscale → threshold → connected components → contours → rooms."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")

    # Black rooms on white background → invert so rooms are white for CC
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    h, w = img.shape[:2]
    candidates: list[dict[str, Any]] = []
    for label in range(1, num_labels):  # skip background 0
        x, y, bw, bh, pixel_area = stats[label]
        area_sqft = float(pixel_area) * (feet_per_pixel**2)
        if area_sqft < min_area_sqft:
            continue
        mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, simplify_epsilon_px, True)
        coords_px = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(coords_px) < 3:
            continue
        # Convert image coords (y down) to plan coords (y up)
        coords = [(x * feet_per_pixel, (h - y) * feet_per_pixel) for x, y in coords_px]
        touching_exterior = bool(x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
        candidates.append(
            {
                "coords": coords,
                "area": area_sqft,
                "bbox": bounding_box(coords),
                "aspect_ratio": aspect_ratio(coords),
                "touching_exterior": touching_exterior,
            }
        )

    candidates = _order_rooms_by_area(candidates)
    rooms: list[Room] = []
    for i, c in enumerate(candidates):
        rtype = _guess_room_type(c["area"], c["aspect_ratio"], c["touching_exterior"], i)
        zone = ZoneType.PRIVATE if rtype in (RoomType.BEDROOM, RoomType.CLOSET) else ZoneType.PUBLIC
        if rtype == RoomType.BATHROOM:
            zone = ZoneType.WET
        rooms.append(
            Room(
                name=f"{rtype.value}_{i+1}",
                room_type=rtype,
                polygon=to_points(c["coords"]),
                zone=zone,
                metadata={
                    "area": c["area"],
                    "aspect_ratio": c["aspect_ratio"],
                    "touching_exterior": c["touching_exterior"],
                    "bbox": c["bbox"],
                },
            )
        )

    # Boundary = image frame
    boundary = Boundary(
        exterior=to_points(
            [
                (0, 0),
                (w * feet_per_pixel, 0),
                (w * feet_per_pixel, h * feet_per_pixel),
                (0, h * feet_per_pixel),
            ]
        )
    )

    plan = FloorPlan(
        name="imported",
        boundary=boundary,
        rooms=rooms,
        metadata={
            "source": "bw_image",
            "feet_per_pixel": feet_per_pixel,
            "image_size": [w, h],
            "room_candidates": len(rooms),
        },
    )

    # Neighbours via adjacency
    adj = build_adjacency(plan)
    plan = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "neighbours": adj,
            }
        }
    )
    return plan


def synthesize_bw_fixture(width_px: int = 200, height_px: int = 160) -> bytes:
    """Create a simple BW test image with black room rectangles."""
    img = np.ones((height_px, width_px), dtype=np.uint8) * 255
    # rooms as black filled rects
    cv2.rectangle(img, (10, 10), (90, 100), 0, -1)  # living
    cv2.rectangle(img, (100, 10), (190, 70), 0, -1)  # bed1
    cv2.rectangle(img, (100, 80), (190, 150), 0, -1)  # bed2
    cv2.rectangle(img, (10, 110), (50, 150), 0, -1)  # bath
    cv2.rectangle(img, (55, 110), (90, 150), 0, -1)  # kitchen
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("encode failed")
    return buf.tobytes()
