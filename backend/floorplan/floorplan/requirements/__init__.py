"""Program / room-requirement builders."""

from .areas import AREA_OPTIONS, area_options_for, name_root
from .centers import (
    center_coverage_score,
    center_sampling_order,
    improve_center_coverage,
    sample_centers,
)
from .context import (
    document_to_plan,
    generate_plan_document,
    run_generation,
    sample_parameters,
)
from .entry import sample_entry_point
from .graph import build_connection_graph
from .grow import grow_rectangles
from .internals import GenerationInternals
from .plan_io import llm_to_plan, plan_document_from_internals, plan_document_id
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
    "build_requirements",
    "center_coverage_score",
    "center_sampling_order",
    "document_to_plan",
    "generate_plan_document",
    "GenerationInternals",
    "grow_rectangles",
    "improve_center_coverage",
    "is_bedroom",
    "llm_to_plan",
    "name_root",
    "pick_selected_areas",
    "plan_document_from_internals",
    "plan_document_id",
    "run_generation",
    "sample_centers",
    "sample_entry_point",
    "sample_parameters",
    "scale_selected_areas",
    "space_names_for",
    "space_role",
]
