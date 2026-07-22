"""Program / room-requirement builders."""

from .areas import AREA_OPTIONS, area_options_for, name_root
from .centers import (
    center_coverage_score,
    center_sampling_order,
    improve_center_coverage,
    sample_centers,
)
from .context import run_generation
from .entry import sample_entry_point
from .graph import build_connection_graph
from .grow import grow_rectangles
from .plan_io import (
    document_to_plan,
    generate_geometry,
    generate_plan_document,
    place_openings,
    plan_document_id,
    sample_into_document,
    seed_document,
)
from .program import (
    is_bedroom,
    pick_selected_areas,
    scale_selected_areas,
    space_names_for,
    space_role,
)

__all__ = [
    "AREA_OPTIONS",
    "area_options_for",
    "build_connection_graph",
    "center_coverage_score",
    "center_sampling_order",
    "document_to_plan",
    "generate_geometry",
    "generate_plan_document",
    "grow_rectangles",
    "improve_center_coverage",
    "is_bedroom",
    "name_root",
    "pick_selected_areas",
    "place_openings",
    "plan_document_id",
    "run_generation",
    "sample_centers",
    "sample_entry_point",
    "sample_into_document",
    "scale_selected_areas",
    "seed_document",
    "space_names_for",
    "space_role",
]
