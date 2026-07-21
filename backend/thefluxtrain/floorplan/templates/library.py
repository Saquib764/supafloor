"""Apartment template library — logical topology, not coordinates."""

from __future__ import annotations

from thefluxtrain.floorplan.schemas.enums import MutationOp, RoomType, ZoneType
from thefluxtrain.floorplan.schemas.models import (
    Boundary,
    FloorPlan,
    Point2D,
    Room,
    RoomRequirement,
    TemplateSpec,
    Wall,
    Door,
    Window,
)
from thefluxtrain.floorplan.geometry.ops import rect, to_points


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
            _req(RoomType.ENTRY, "Entry", min_area=1.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living/Sleep", min_area=16, preferred_area=22, requires_exterior=True, min_width=3.0),
            _req(RoomType.KITCHEN, "Kitchen", min_area=4, preferred_area=6, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.BATHROOM, "Bathroom", min_area=3.5, preferred_zone=ZoneType.WET, min_width=1.4),
            _req(RoomType.CLOSET, "Closet", required=False, min_area=1.0, preferred_zone=ZoneType.PRIVATE),
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
            _req(RoomType.ENTRY, "Entry", min_area=1.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=14, preferred_area=18, requires_exterior=True, min_width=3.2),
            _req(RoomType.KITCHEN, "Kitchen", min_area=5, preferred_area=7, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.DINING, "Dining", required=False, min_area=6, preferred_area=8),
            _req(RoomType.HALLWAY, "Hall", min_area=2, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.BEDROOM, "Bedroom 1", min_area=9, preferred_area=11, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=2.6),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=8, preferred_area=10, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=2.4),
            _req(RoomType.BATHROOM, "Bathroom", min_area=3.5, preferred_zone=ZoneType.WET, min_width=1.4),
            _req(RoomType.BALCONY, "Balcony", required=False, min_area=3, preferred_zone=ZoneType.OUTDOOR, requires_exterior=True),
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
            _req(RoomType.ENTRY, "Entry", min_area=2, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=20, preferred_area=28, requires_exterior=True, min_width=3.6),
            _req(RoomType.DINING, "Dining", min_area=8, preferred_area=12),
            _req(RoomType.KITCHEN, "Kitchen", min_area=8, preferred_area=12, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=14, preferred_area=18, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=3.0),
            _req(RoomType.BATHROOM, "Master Bath", min_area=5, preferred_zone=ZoneType.WET, min_width=1.6),
            _req(RoomType.CLOSET, "Walk-in Closet", min_area=3, preferred_zone=ZoneType.PRIVATE),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=10, preferred_area=12, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=2.6),
            _req(RoomType.POWDER, "Powder", min_area=2, preferred_zone=ZoneType.WET),
            _req(RoomType.BALCONY, "Balcony", min_area=5, preferred_zone=ZoneType.OUTDOOR, requires_exterior=True),
            _req(RoomType.UTILITY, "Utility", required=False, min_area=2, preferred_zone=ZoneType.SERVICE),
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
            _req(RoomType.ENTRY, "Entry", min_area=2, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=16, preferred_area=22, requires_exterior=True, min_width=3.4),
            _req(RoomType.KITCHEN, "Kitchen", min_area=7, preferred_area=9, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.DINING, "Dining", min_area=7, preferred_area=9),
            _req(RoomType.HALLWAY, "Hall", min_area=3, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=12, preferred_area=15, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=2.8),
            _req(RoomType.BATHROOM, "Master Bath", min_area=4, preferred_zone=ZoneType.WET),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=9, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=2.5),
            _req(RoomType.BEDROOM, "Bedroom 3", min_area=8, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=2.4),
            _req(RoomType.BATHROOM, "Bath 2", min_area=3.5, preferred_zone=ZoneType.WET),
            _req(RoomType.BALCONY, "Balcony", required=False, min_area=4, preferred_zone=ZoneType.OUTDOOR),
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
            _req(RoomType.ENTRY, "Entry", min_area=2.5, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=20, preferred_area=28, requires_exterior=True, min_width=3.6),
            _req(RoomType.DINING, "Dining", min_area=9, preferred_area=12),
            _req(RoomType.KITCHEN, "Kitchen", min_area=9, preferred_area=12, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.HALLWAY, "Hall", min_area=4, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=14, preferred_zone=ZoneType.PRIVATE, requires_exterior=True, min_width=3.0),
            _req(RoomType.BATHROOM, "Master Bath", min_area=5, preferred_zone=ZoneType.WET),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=10, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BEDROOM, "Bedroom 3", min_area=9, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BEDROOM, "Bedroom 4", min_area=9, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BATHROOM, "Bath 2", min_area=4, preferred_zone=ZoneType.WET),
            _req(RoomType.UTILITY, "Utility", min_area=2.5, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.BALCONY, "Balcony", min_area=5, preferred_zone=ZoneType.OUTDOOR),
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
            _req(RoomType.ENTRY, "Entry", min_area=2, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.LIVING, "Living", min_area=18, preferred_area=24, requires_exterior=True),
            _req(RoomType.KITCHEN, "Kitchen", min_area=8, preferred_zone=ZoneType.SERVICE),
            _req(RoomType.DINING, "Dining", min_area=8),
            _req(RoomType.POWDER, "Powder", min_area=2, preferred_zone=ZoneType.WET),
            _req(RoomType.HALLWAY, "Upper Hall", min_area=3, preferred_zone=ZoneType.CIRCULATION),
            _req(RoomType.MASTER_BEDROOM, "Master Bedroom", min_area=14, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BATHROOM, "Master Bath", min_area=5, preferred_zone=ZoneType.WET),
            _req(RoomType.BEDROOM, "Bedroom 2", min_area=10, preferred_zone=ZoneType.PRIVATE, requires_exterior=True),
            _req(RoomType.BATHROOM, "Bath 2", min_area=3.5, preferred_zone=ZoneType.WET),
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
    return Room(name=name, room_type=room_type, polygon=to_points(coords), zone=zone)


def instantiate_compact_2bed(width: float = 10.0, depth: float = 8.0) -> FloorPlan:
    """Deterministic rectangular compact 2BR layout fitted to a box boundary."""
    w, d = width, depth
    # Grid proportions (normalized then scaled)
    # Living (left), kitchen (top-right of living), hall+beds (right)
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

    # Simple walls along room edges (exterior flag for boundary)
    walls = [
        Wall(start=Point2D(x=0, y=0), end=Point2D(x=w, y=0), is_exterior=True),
        Wall(start=Point2D(x=w, y=0), end=Point2D(x=w, y=d), is_exterior=True),
        Wall(start=Point2D(x=w, y=d), end=Point2D(x=0, y=d), is_exterior=True),
        Wall(start=Point2D(x=0, y=d), end=Point2D(x=0, y=0), is_exterior=True),
    ]

    id_map = {r.name: r.id for r in rooms}
    doors = [
        Door(
            wall_id=walls[0].id,
            position=Point2D(x=w * 0.52, y=0),
            connects=[id_map["Entry"]],
            is_exterior=True,
        ),
        Door(
            wall_id="internal",
            position=Point2D(x=w * 0.45, y=d * 0.15),
            connects=[id_map["Entry"], id_map["Living"]],
        ),
        Door(
            wall_id="internal",
            position=Point2D(x=w * 0.45, y=d * 0.40),
            connects=[id_map["Living"], id_map["Hall"]],
        ),
        Door(
            wall_id="internal",
            position=Point2D(x=w * 0.52, y=d * 0.50),
            connects=[id_map["Hall"], id_map["Bedroom 1"]],
        ),
        Door(
            wall_id="internal",
            position=Point2D(x=w * 0.52, y=d * 0.65),
            connects=[id_map["Hall"], id_map["Bedroom 2"]],
        ),
        Door(
            wall_id="internal",
            position=Point2D(x=w * 0.52, y=d * 0.80),
            connects=[id_map["Hall"], id_map["Bathroom"]],
        ),
        Door(
            wall_id="internal",
            position=Point2D(x=w * 0.22, y=d * 0.70),
            connects=[id_map["Living"], id_map["Kitchen"]],
        ),
    ]

    windows = [
        Window(wall_id=walls[2].id, position=Point2D(x=w * 0.22, y=d), width=1.5),
        Window(wall_id=walls[1].id, position=Point2D(x=w, y=d * 0.25), width=1.2),
        Window(wall_id=walls[1].id, position=Point2D(x=w, y=d * 0.75), width=1.2),
        Window(wall_id=walls[3].id, position=Point2D(x=0, y=d * 0.35), width=1.4),
    ]

    return FloorPlan(
        name="Compact 2 Bedroom",
        boundary=boundary,
        rooms=rooms,
        walls=walls,
        doors=doors,
        windows=windows,
        metadata={"template_id": "compact_2bed"},
    )


def instantiate_studio(width: float = 7.0, depth: float = 5.5) -> FloorPlan:
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
    walls = [
        Wall(start=Point2D(x=0, y=0), end=Point2D(x=w, y=0), is_exterior=True),
        Wall(start=Point2D(x=w, y=0), end=Point2D(x=w, y=d), is_exterior=True),
        Wall(start=Point2D(x=w, y=d), end=Point2D(x=0, y=d), is_exterior=True),
        Wall(start=Point2D(x=0, y=d), end=Point2D(x=0, y=0), is_exterior=True),
    ]
    id_map = {r.name: r.id for r in rooms}
    doors = [
        Door(wall_id=walls[0].id, position=Point2D(x=w * 0.2, y=0), connects=[id_map["Living/Sleep"]], is_exterior=True),
        Door(wall_id="internal", position=Point2D(x=w * 0.70, y=d * 0.20), connects=[id_map["Living/Sleep"], id_map["Bathroom"]]),
        Door(wall_id="internal", position=Point2D(x=w * 0.70, y=d * 0.55), connects=[id_map["Living/Sleep"], id_map["Kitchen"]]),
    ]
    windows = [
        Window(wall_id=walls[2].id, position=Point2D(x=w * 0.35, y=d), width=1.8),
        Window(wall_id=walls[3].id, position=Point2D(x=0, y=d * 0.5), width=1.2),
    ]
    return FloorPlan(
        name="Studio",
        boundary=boundary,
        rooms=rooms,
        walls=walls,
        doors=doors,
        windows=windows,
        metadata={"template_id": "studio"},
    )


def instantiate_3bed(width: float = 12.0, depth: float = 9.0) -> FloorPlan:
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
    walls = [
        Wall(start=Point2D(x=0, y=0), end=Point2D(x=w, y=0), is_exterior=True),
        Wall(start=Point2D(x=w, y=0), end=Point2D(x=w, y=d), is_exterior=True),
        Wall(start=Point2D(x=w, y=d), end=Point2D(x=0, y=d), is_exterior=True),
        Wall(start=Point2D(x=0, y=d), end=Point2D(x=0, y=0), is_exterior=True),
    ]
    id_map = {r.name: r.id for r in rooms}
    doors = [
        Door(wall_id=walls[0].id, position=Point2D(x=w * 0.46, y=0), connects=[id_map["Entry"]], is_exterior=True),
        Door(wall_id="i", position=Point2D(x=w * 0.40, y=d * 0.10), connects=[id_map["Entry"], id_map["Living"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.40, y=d * 0.35), connects=[id_map["Living"], id_map["Hall"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.11, y=d * 0.55), connects=[id_map["Living"], id_map["Dining"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.22, y=d * 0.75), connects=[id_map["Dining"], id_map["Kitchen"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.52, y=d * 0.30), connects=[id_map["Hall"], id_map["Master Bedroom"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.60, y=d * 0.40), connects=[id_map["Master Bedroom"], id_map["Master Bath"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.72, y=d * 0.50), connects=[id_map["Hall"], id_map["Bedroom 2"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.60, y=d * 0.60), connects=[id_map["Hall"], id_map["Bedroom 3"]]),
        Door(wall_id="i", position=Point2D(x=w * 0.46, y=d * 0.80), connects=[id_map["Hall"], id_map["Bath 2"]]),
    ]
    windows = [
        Window(wall_id=walls[3].id, position=Point2D(x=0, y=d * 0.25), width=1.5),
        Window(wall_id=walls[2].id, position=Point2D(x=w * 0.10, y=d), width=1.2),
        Window(wall_id=walls[1].id, position=Point2D(x=w, y=d * 0.20), width=1.4),
        Window(wall_id=walls[1].id, position=Point2D(x=w, y=d * 0.55), width=1.2),
        Window(wall_id=walls[1].id, position=Point2D(x=w, y=d * 0.80), width=1.2),
    ]
    return FloorPlan(
        name="3 Bedroom",
        boundary=boundary,
        rooms=rooms,
        walls=walls,
        doors=doors,
        windows=windows,
        metadata={"template_id": "3bed"},
    )


INSTANTIATORS = {
    "studio": instantiate_studio,
    "compact_2bed": instantiate_compact_2bed,
    "luxury_2bed": lambda w=12.0, d=9.5: instantiate_3bed(w, d).model_copy(
        update={"name": "Luxury 2 Bedroom", "metadata": {"template_id": "luxury_2bed"}}
    ),
    "3bed": instantiate_3bed,
    "3bed_corner": lambda w=12.0, d=9.0: instantiate_3bed(w, d).model_copy(
        update={"name": "3 Bedroom Corner", "metadata": {"template_id": "3bed_corner"}}
    ),
    "4bed": lambda w=14.0, d=10.0: instantiate_3bed(w, d).model_copy(
        update={"name": "4 Bedroom", "metadata": {"template_id": "4bed"}}
    ),
    "duplex": lambda w=10.0, d=8.0: instantiate_compact_2bed(w, d).model_copy(
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
    # Some lambdas use w/d
    try:
        return fn(**kwargs) if kwargs else fn()
    except TypeError:
        # luxury etc use positional via defaults
        if width is not None and depth is not None:
            return fn(width, depth)
        return fn()
