"""Program / room-requirement builders."""

from .areas import AREA_OPTIONS, area_options_for, name_root
from .centers import (
    center_coverage_score,
    center_sampling_order,
    improve_center_coverage,
    sample_centers,
)
from .context import (
    build_document,
    document_to_plan,
    run_generation,
    sample_parameters,
)
from .entry import sample_entry_point
from .graph import build_connection_graph
from .grow import grow_rectangles
from .plan_io import internal_to_plan_document, llm_to_plan, plan_document_id
from .program import (
    build_requirements,
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
    "build_document",
    "build_requirements",
    "center_coverage_score",
    "center_sampling_order",
    "document_to_plan",
    "grow_rectangles",
    "improve_center_coverage",
    "internal_to_plan_document",
    "is_bedroom",
    "llm_to_plan",
    "name_root",
    "pick_selected_areas",
    "plan_document_id",
    "run_generation",
    "sample_centers",
    "sample_entry_point",
    "sample_parameters",
    "scale_selected_areas",
    "space_names_for",
    "space_role",
]
