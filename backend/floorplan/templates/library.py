"""Apartment template library — logical topology; sizes in feet / sqft."""

from __future__ import annotations

from floorplan.schemas.enums import MutationOp, RoomType, ZoneType
from floorplan.schemas.models import (
    Boundary,
    Door,
    FloorPlan,
    Room,
    RoomRequirement,
    TemplateSpec,
    Window,
)
from floorplan.geometry.ops import rect, to_points
from floorplan.geometry.authors import (
    horizontal_opening,
    make_door_line,
    make_room,
    make_window_line,
    vertical_opening,
)


def _req(
    room_type: RoomType,
    name: str,
    *,
    required: bool = True,
    min_area: float = 0.0,
    preferred_area: float | None = None,
    preferred_zone: ZoneType = ZoneType.PUBLIC,
    requires_exterior: bool = False,
    min_width: float = 0.0,
) -> RoomRequirement:
    return RoomRequirement(
        room_type=room_type,
        name=name,
        required=required,
        min_area=min_area,
        preferred_area=preferred_area,
        preferred_zone=preferred_zone,
        requires_exterior=requires_exterior,
        min_width=min_width,
    )


COMMON_MUTATIONS = [
    MutationOp.MOVE_WALL,
    MutationOp.RESIZE_ROOM,
    MutationOp.SWAP_ROOMS,
    MutationOp.INSERT_CLOSET,
    MutationOp.MIRROR,
    MutationOp.EXPAND_LIVING,
    MutationOp.REDUCE_HALLWAY,
]


def studio_template() -> TemplateSpec:
    return TemplateSpec(
        id="studio",
        name="Studio",
        description="Compact studio with open living/sleeping and wet core.",
        rooms=[
            _req(RoomType.ENTRY, "Entry", min_area=16.1, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living/Sleep", min_area=172.2, preferred_area=236.8, requires_exterior=True, min_width=9.84),
            _req(RoomType.KITCHEN, "Kitchen", min_area=43.1, preferred_area=64.6, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.BATHROOM, "Bathroom", min_area=37.7, preferred_zone=ZoneType.WET, min_width=4.59),
            _req(RoomType.CLOSET, "Closet", required=False, min_area=10.8, preferred_zone=ZoneType.PRIVATE),
        ],
        preferred_adjacency=[
            ("Entry", "Living/Sleep"),
            ("Living/Sleep", "Kitchen"),
            ("Living/Sleep", "Bathroom"),
        ],
        required_exterior=["Living/Sleep"],
        wet_core_rooms=["Bathroom", "Kitchen"],
        expansion_points=["Living/Sleep"],
        mutation_rules=COMMON_MUTATIONS + [MutationOp.INSERT_BALCONY],
    )


def compact_2bed_template() -> TemplateSpec:
    return TemplateSpec(
        id="compact_2bed",
        name="Compact 2 Bedroom",
        description="Efficient 2BR with shared bath and compact wet core.",
        rooms=[
            _req(RoomType.ENTRY, "Entry", min_area=16.1, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=150.7, preferred_area=193.8, requires_exterior=True, min_width=10.5),
            _req(RoomType.KITCHEN, "Kitchen", min_area=53.8, preferred_area=75.3, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.DINING, "Dining", required=False, min_area=64.6, preferred_area=86.1),
            _req(RoomType.HALLWAY, "Hall", min_area=21.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.BEDROOM, "Bedroom 1", min_area=96.9, preferred_area=118.4, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=8.53),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=86.1, preferred_area=107.6, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=7.87),
            _req(RoomType.BATHROOM, "Bathroom", min_area=37.7, preferred_zone=ZoneType.WET, min_width=4.59),
            _req(RoomType.BALCONY, "Balcony", required=False, min_area=32.3, preferred_zone=ZoneType.OUTDOOR, requires_exterior=True),
        ],
        preferred_adjacency=[
            ("Entry", "Living"),
            ("Living", "Kitchen"),
            ("Living", "Hall"),
            ("Hall", "Bedroom 1"),
            ("Hall", "Bedroom 2"),
            ("Hall", "Bathroom"),
            ("Living", "Balcony"),
        ],
        required_exterior=["Living", "Bedroom 1", "Bedroom 2"],
        wet_core_rooms=["Bathroom", "Kitchen"],
        expansion_points=["Living", "Bedroom 1"],
        mutation_rules=COMMON_MUTATIONS + [MutationOp.INSERT_BALCONY, MutationOp.INSERT_TOILET],
    )


def luxury_2bed_template() -> TemplateSpec:
    return TemplateSpec(
        id="luxury_2bed",
        name="Luxury 2 Bedroom",
        description="2BR with master ensuite, powder room, and open living.",
        rooms=[
            _req(RoomType.ENTRY, "Entry", min_area=21.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=215.3, preferred_area=301.4, requires_exterior=True, min_width=11.81),
            _req(RoomType.DINING, "Dining", min_area=86.1, preferred_area=129.2),
            _req(RoomType.KITCHEN, "Kitchen", min_area=86.1, preferred_area=129.2, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=150.7, preferred_area=193.8, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=9.84),
            _req(RoomType.BATHROOM, "Master Bath", min_area=53.8, preferred_zone=ZoneType.WET, min_width=5.25),
            _req(RoomType.CLOSET, "Walk-in Closet", min_area=32.3, preferred_zone=ZoneType.PRIVATE),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=107.6, preferred_area=129.2, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=8.53),
            _req(RoomType.POWDER, "Powder", min_area=21.5, preferred_zone=ZoneType.WET),
            _req(RoomType.BALCONY, "Balcony", min_area=53.8, preferred_zone=ZoneType.OUTDOOR, requires_exterior=True),
            _req(RoomType.UTILITY, "Utility", required=False, min_area=21.5, preferred_zone=ZoneType.SERVICE),
        ],
        preferred_adjacency=[
            ("Entry", "Living"),
            ("Living", "Dining"),
            ("Dining", "Kitchen"),
            ("Living", "Balcony"),
            ("Master Bedroom", "Master Bath"),
            ("Master Bedroom", "Walk-in Closet"),
        ],
        required_exterior=["Living", "Master Bedroom", "Bedroom 2", "Balcony"],
        wet_core_rooms=["Master Bath", "Powder", "Kitchen", "Utility"],
        expansion_points=["Living", "Master Bedroom"],
        mutation_rules=COMMON_MUTATIONS + [MutationOp.INSERT_STORAGE, MutationOp.SHIFT_WET_CORE],
    )


def three_bed_template() -> TemplateSpec:
    return TemplateSpec(
        id="3bed",
        name="3 Bedroom",
        description="Family 3BR with living, kitchen, and two baths.",
        rooms=[
            _req(RoomType.ENTRY, "Entry", min_area=21.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=172.2, preferred_area=236.8, requires_exterior=True, min_width=11.15),
            _req(RoomType.KITCHEN, "Kitchen", min_area=75.3, preferred_area=96.9, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.DINING, "Dining", min_area=75.3, preferred_area=96.9),
            _req(RoomType.HALLWAY, "Hall", min_area=32.3, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=129.2, preferred_area=161.5, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=9.19),
            _req(RoomType.BATHROOM, "Master Bath", min_area=43.1, preferred_zone=ZoneType.WET),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=96.9, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=8.2),
            _req(RoomType.BEDROOM, "Bedroom 3", min_area=86.1, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=7.87),
            _req(RoomType.BATHROOM, "Bath 2", min_area=37.7, preferred_zone=ZoneType.WET),
            _req(RoomType.BALCONY, "Balcony", required=False, min_area=43.1, preferred_zone=ZoneType.OUTDOOR),
        ],
        preferred_adjacency=[
            ("Entry", "Living"),
            ("Living", "Dining"),
            ("Dining", "Kitchen"),
            ("Living", "Hall"),
            ("Hall", "Master Bedroom"),
            ("Hall", "Bedroom 2"),
            ("Hall", "Bedroom 3"),
            ("Master Bedroom", "Master Bath"),
            ("Hall", "Bath 2"),
        ],
        required_exterior=["Living", "Master Bedroom", "Bedroom 2", "Bedroom 3"],
        wet_core_rooms=["Master Bath", "Bath 2", "Kitchen"],
        expansion_points=["Living", "Master Bedroom"],
        mutation_rules=COMMON_MUTATIONS + [MutationOp.INSERT_CLOSET, MutationOp.SPLIT_ROOM],
    )


def three_bed_corner_template() -> TemplateSpec:
    t = three_bed_template().model_copy(
        update={
            "id": "3bed_corner",
            "name": "3 Bedroom Corner",
            "description": "Corner 3BR with dual-aspect living and more exterior exposure.",
            "required_exterior": ["Living", "Master Bedroom", "Bedroom 2", "Bedroom 3", "Kitchen"],
        }
    )
    return t


def four_bed_template() -> TemplateSpec:
    return TemplateSpec(
        id="4bed",
        name="4 Bedroom",
        description="Large family 4BR layout.",
        rooms=[
            _req(RoomType.ENTRY, "Entry", min_area=26.9, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=215.3, preferred_area=301.4, requires_exterior=True, min_width=11.81),
            _req(RoomType.DINING, "Dining", min_area=96.9, preferred_area=129.2),
            _req(RoomType.KITCHEN, "Kitchen", min_area=96.9, preferred_area=129.2, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.HALLWAY, "Hall", min_area=43.1, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=150.7, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=9.84),
            _req(RoomType.BATHROOM, "Master Bath", min_area=53.8, preferred_zone=ZoneType.WET),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=107.6, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BEDROOM, "Bedroom 3", min_area=96.9, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BEDROOM, "Bedroom 4", min_area=96.9, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BATHROOM, "Bath 2", min_area=43.1, preferred_zone=ZoneType.WET),
            _req(RoomType.UTILITY, "Utility", min_area=26.9, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.BALCONY, "Balcony", min_area=53.8, preferred_zone=ZoneType.OUTDOOR),
        ],
        preferred_adjacency=[
            ("Entry", "Living"),
            ("Living", "Dining"),
            ("Dining", "Kitchen"),
            ("Living", "Hall"),
            ("Hall", "Master Bedroom"),
            ("Hall", "Bedroom 2"),
            ("Hall", "Bedroom 3"),
            ("Hall", "Bedroom 4"),
        ],
        required_exterior=["Living", "Master Bedroom", "Bedroom 2", "Bedroom 3", "Bedroom 4"],
        wet_core_rooms=["Master Bath", "Bath 2", "Kitchen", "Utility"],
        expansion_points=["Living"],
        mutation_rules=COMMON_MUTATIONS + [MutationOp.MERGE_ROOMS, MutationOp.INSERT_STORAGE],
    )


def duplex_template() -> TemplateSpec:
    return TemplateSpec(
        id="duplex",
        name="Duplex",
        description="Two-level duplex: public below, private above (logical).",
        rooms=[
            _req(RoomType.ENTRY, "Entry", min_area=21.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=193.8, preferred_area=258.3, requires_exterior=True),
            _req(RoomType.KITCHEN, "Kitchen", min_area=86.1, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.DINING, "Dining", min_area=86.1),
            _req(RoomType.POWDER, "Powder", min_area=21.5, preferred_zone=ZoneType.WET),
            _req(RoomType.HALLWAY, "Upper Hall", min_area=32.3, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=150.7, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BATHROOM, "Master Bath", min_area=53.8, preferred_zone=ZoneType.WET),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=107.6, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BATHROOM, "Bath 2", min_area=37.7, preferred_zone=ZoneType.WET),
        ],
        preferred_adjacency=[
            ("Entry", "Living"),
            ("Living", "Dining"),
            ("Dining", "Kitchen"),
            ("Living", "Powder"),
            ("Upper Hall", "Master Bedroom"),
            ("Upper Hall", "Bedroom 2"),
        ],
        required_exterior=["Living", "Master Bedroom", "Bedroom 2"],
        wet_core_rooms=["Powder", "Master Bath", "Bath 2", "Kitchen"],
        expansion_points=["Living", "Master Bedroom"],
        mutation_rules=COMMON_MUTATIONS,
    )


TEMPLATES: dict[str, TemplateSpec] = {
    t.id: t
    for t in [
        studio_template(),
        compact_2bed_template(),
        luxury_2bed_template(),
        three_bed_template(),
        three_bed_corner_template(),
        four_bed_template(),
        duplex_template(),
    ]
}


def get_template(template_id: str) -> TemplateSpec:
    if template_id not in TEMPLATES:
        raise KeyError(f"Unknown template: {template_id}. Available: {list(TEMPLATES)}")
    return TEMPLATES[template_id]


def list_templates() -> list[dict]:
    return [
        {"id": t.id, "name": t.name, "description": t.description, "room_count": len(t.rooms)}
        for t in TEMPLATES.values()
    ]


def _room(name: str, room_type: RoomType, coords: list[tuple[float, float]], zone: ZoneType) -> Room:
    return make_room(name, coords, room_type=room_type, zone=zone)


def _hdoor(xc: float, y: float, length: float, connects: list[str], *, exterior: bool = False) -> Door:
    a, b = horizontal_opening(xc, y, length)
    return make_door_line(a, b, connects=connects, is_exterior=exterior)


def _vdoor(x: float, yc: float, length: float, connects: list[str], *, exterior: bool = False) -> Door:
    a, b = vertical_opening(x, yc, length)
    return make_door_line(a, b, connects=connects, is_exterior=exterior)


def _hwin(xc: float, y: float, length: float, room_ids: list[str] | None = None) -> Window:
    a, b = horizontal_opening(xc, y, length)
    return make_window_line(a, b, room_ids=room_ids)


def _vwin(x: float, yc: float, length: float, room_ids: list[str] | None = None) -> Window:
    a, b = vertical_opening(x, yc, length)
    return make_window_line(a, b, room_ids=room_ids)


def instantiate_compact_2bed(width: float = 32.81, depth: float = 26.25) -> FloorPlan:
    """Deterministic rectangular compact 2BR: rooms + door lines + window lines."""
    w, d = width, depth
    living = rect(0, 0, w * 0.45, d * 0.70)
    kitchen = rect(0, d * 0.70, w * 0.45, d * 0.30)
    entry = rect(w * 0.45, 0, w * 0.15, d * 0.25)
    hall = rect(w * 0.45, d * 0.25, w * 0.15, d * 0.45)
    bath = rect(w * 0.45, d * 0.70, w * 0.15, d * 0.30)
    bed1 = rect(w * 0.60, 0, w * 0.40, d * 0.50)
    bed2 = rect(w * 0.60, d * 0.50, w * 0.40, d * 0.50)

    rooms = [
        _room("Living", RoomType.LIVING, living, ZoneType.PUBLIC),
        _room("Kitchen", RoomType.KITCHEN, kitchen, ZoneType.SERVICE),
        _room("Entry", RoomType.ENTRY, entry, ZoneType.CIRCULATION),
        _room("Hall", RoomType.HALLWAY, hall, ZoneType.CIRCULATION),
        _room("Bathroom", RoomType.BATHROOM, bath, ZoneType.WET),
        _room("Bedroom 1", RoomType.BEDROOM, bed1, ZoneType.PRIVATE),
        _room("Bedroom 2", RoomType.BEDROOM, bed2, ZoneType.PRIVATE),
    ]

    boundary = Boundary(exterior=to_points(rect(0, 0, w, d)))
    id_map = {r.name: r.id for r in rooms}

    doors = [
        _hdoor(w * 0.52, 0.0, 2.95, [id_map["Entry"]], exterior=True),
        _vdoor(w * 0.45, d * 0.12, 2.95, [id_map["Entry"], id_map["Living"]]),
        _vdoor(w * 0.45, d * 0.40, 2.95, [id_map["Living"], id_map["Hall"]]),
        _vdoor(w * 0.60, d * 0.25, 2.95, [id_map["Hall"], id_map["Bedroom 1"]]),
        _vdoor(w * 0.60, d * 0.65, 2.95, [id_map["Hall"], id_map["Bedroom 2"]]),
        _vdoor(w * 0.60, d * 0.80, 2.62, [id_map["Hall"], id_map["Bathroom"]]),
        _hdoor(w * 0.22, d * 0.70, 2.95, [id_map["Living"], id_map["Kitchen"]]),
    ]

    windows = [
        _hwin(w * 0.22, d, 4.92, [id_map["Kitchen"]]),
        _vwin(w, d * 0.25, 3.94, [id_map["Bedroom 1"]]),
        _vwin(w, d * 0.75, 3.94, [id_map["Bedroom 2"]]),
        _vwin(0.0, d * 0.35, 4.59, [id_map["Living"]]),
    ]

    return FloorPlan(
        name="Compact 2 Bedroom",
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        metadata={"template_id": "compact_2bed"},
    )


def instantiate_studio(width: float = 22.97, depth: float = 18.04) -> FloorPlan:
    w, d = width, depth
    living = rect(0, 0, w * 0.70, d)
    kitchen = rect(w * 0.70, d * 0.40, w * 0.30, d * 0.60)
    bath = rect(w * 0.70, 0, w * 0.30, d * 0.40)
    rooms = [
        _room("Living/Sleep", RoomType.LIVING, living, ZoneType.PUBLIC),
        _room("Kitchen", RoomType.KITCHEN, kitchen, ZoneType.SERVICE),
        _room("Bathroom", RoomType.BATHROOM, bath, ZoneType.WET),
    ]
    boundary = Boundary(exterior=to_points(rect(0, 0, w, d)))
    id_map = {r.name: r.id for r in rooms}
    doors = [
        _hdoor(w * 0.20, 0.0, 2.95, [id_map["Living/Sleep"]], exterior=True),
        _vdoor(w * 0.70, d * 0.20, 2.62, [id_map["Living/Sleep"], id_map["Bathroom"]]),
        _vdoor(w * 0.70, d * 0.55, 2.95, [id_map["Living/Sleep"], id_map["Kitchen"]]),
    ]
    windows = [
        _hwin(w * 0.35, d, 5.91, [id_map["Living/Sleep"]]),
        _vwin(0.0, d * 0.50, 3.94, [id_map["Living/Sleep"]]),
    ]
    return FloorPlan(
        name="Studio",
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        metadata={"template_id": "studio"},
    )


def instantiate_3bed(width: float = 39.37, depth: float = 29.53) -> FloorPlan:
    w, d = width, depth
    living = rect(0, 0, w * 0.40, d * 0.55)
    dining = rect(0, d * 0.55, w * 0.22, d * 0.45)
    kitchen = rect(w * 0.22, d * 0.55, w * 0.18, d * 0.45)
    entry = rect(w * 0.40, 0, w * 0.12, d * 0.22)
    hall = rect(w * 0.40, d * 0.22, w * 0.12, d * 0.53)
    bath2 = rect(w * 0.40, d * 0.75, w * 0.12, d * 0.25)
    master = rect(w * 0.52, 0, w * 0.48, d * 0.40)
    mbath = rect(w * 0.52, d * 0.40, w * 0.20, d * 0.20)
    bed2 = rect(w * 0.72, d * 0.40, w * 0.28, d * 0.30)
    bed3 = rect(w * 0.52, d * 0.60, w * 0.48, d * 0.40)

    rooms = [
        _room("Living", RoomType.LIVING, living, ZoneType.PUBLIC),
        _room("Dining", RoomType.DINING, dining, ZoneType.PUBLIC),
        _room("Kitchen", RoomType.KITCHEN, kitchen, ZoneType.SERVICE),
        _room("Entry", RoomType.ENTRY, entry, ZoneType.CIRCULATION),
        _room("Hall", RoomType.HALLWAY, hall, ZoneType.CIRCULATION),
        _room("Bath 2", RoomType.BATHROOM, bath2, ZoneType.WET),
        _room("Master Bedroom", RoomType.MASTER_BEDROOM, master, ZoneType.PRIVATE),
        _room("Master Bath", RoomType.BATHROOM, mbath, ZoneType.WET),
        _room("Bedroom 2", RoomType.BEDROOM, bed2, ZoneType.PRIVATE),
        _room("Bedroom 3", RoomType.BEDROOM, bed3, ZoneType.PRIVATE),
    ]
    boundary = Boundary(exterior=to_points(rect(0, 0, w, d)))
    id_map = {r.name: r.id for r in rooms}
    doors = [
        _hdoor(w * 0.46, 0.0, 2.95, [id_map["Entry"]], exterior=True),
        _vdoor(w * 0.40, d * 0.10, 2.95, [id_map["Entry"], id_map["Living"]]),
        _vdoor(w * 0.40, d * 0.35, 2.95, [id_map["Living"], id_map["Hall"]]),
        _hdoor(w * 0.11, d * 0.55, 2.95, [id_map["Living"], id_map["Dining"]]),
        _vdoor(w * 0.22, d * 0.75, 2.95, [id_map["Dining"], id_map["Kitchen"]]),
        _vdoor(w * 0.52, d * 0.25, 2.95, [id_map["Hall"], id_map["Master Bedroom"]]),
        _hdoor(w * 0.60, d * 0.40, 2.62, [id_map["Master Bedroom"], id_map["Master Bath"]]),
        _vdoor(w * 0.72, d * 0.50, 2.95, [id_map["Hall"], id_map["Bedroom 2"]]),
        _hdoor(w * 0.60, d * 0.60, 2.95, [id_map["Hall"], id_map["Bedroom 3"]]),
        _vdoor(w * 0.52, d * 0.85, 2.62, [id_map["Hall"], id_map["Bath 2"]]),
    ]
    windows = [
        _vwin(0.0, d * 0.25, 4.92, [id_map["Living"]]),
        _hwin(w * 0.10, d, 3.94, [id_map["Dining"]]),
        _vwin(w, d * 0.20, 4.59, [id_map["Master Bedroom"]]),
        _vwin(w, d * 0.55, 3.94, [id_map["Bedroom 2"]]),
        _vwin(w, d * 0.80, 3.94, [id_map["Bedroom 3"]]),
    ]
    return FloorPlan(
        name="3 Bedroom",
        rooms=rooms,
        doors=doors,
        windows=windows,
        boundary=boundary,
        metadata={"template_id": "3bed"},
    )


INSTANTIATORS = {
    "studio": instantiate_studio,
    "compact_2bed": instantiate_compact_2bed,
    "luxury_2bed": lambda w=39.37, d=31.17: instantiate_3bed(w, d).model_copy(
        update={"name": "Luxury 2 Bedroom", "metadata": {"template_id": "luxury_2bed"}}
    ),
    "3bed": instantiate_3bed,
    "3bed_corner": lambda w=39.37, d=29.53: instantiate_3bed(w, d).model_copy(
        update={"name": "3 Bedroom Corner", "metadata": {"template_id": "3bed_corner"}}
    ),
    "4bed": lambda w=45.93, d=32.81: instantiate_3bed(w, d).model_copy(
        update={"name": "4 Bedroom", "metadata": {"template_id": "4bed"}}
    ),
    "duplex": lambda w=32.81, d=26.25: instantiate_compact_2bed(w, d).model_copy(
        update={"name": "Duplex", "metadata": {"template_id": "duplex"}}
    ),
}


def instantiate_template(template_id: str, width: float | None = None, depth: float | None = None) -> FloorPlan:
    fn = INSTANTIATORS.get(template_id)
    if fn is None:
        raise KeyError(f"No instantiator for template: {template_id}")
    kwargs = {}
    if width is not None:
        kwargs["width"] = width
    if depth is not None:
        kwargs["depth"] = depth
    try:
        return fn(**kwargs) if kwargs else fn()
    except TypeError:
        if width is not None and depth is not None:
            return fn(width, depth)
        return fn()
