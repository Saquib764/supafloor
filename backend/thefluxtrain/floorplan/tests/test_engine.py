"""Template, mutation, evaluation, render, import, dataset tests."""

from thefluxtrain.floorplan.dataset import generate_dataset, generate_sample
from thefluxtrain.floorplan.evaluator import evaluate_rules, score_plan
from thefluxtrain.floorplan.importer import import_bw_image, synthesize_bw_fixture
from thefluxtrain.floorplan.mutations import apply_mutation, mutate_pipeline
from thefluxtrain.floorplan.renderer import render_plan, render_svg
from thefluxtrain.floorplan.schemas.enums import MutationOp
from thefluxtrain.floorplan.services import FloorPlanEngine
from thefluxtrain.floorplan.services.boundary_adapt import adapt_to_boundary, generate_boundary
from thefluxtrain.floorplan.templates import get_template, instantiate_template, list_templates
from thefluxtrain.floorplan.topology import build_adjacency, topology_hash


def test_templates_listed():
    templates = list_templates()
    ids = {t["id"] for t in templates}
    assert "studio" in ids
    assert "compact_2bed" in ids
    assert "3bed" in ids
    assert "4bed" in ids
    assert "duplex" in ids
    t = get_template("compact_2bed")
    assert t.rooms
    assert t.preferred_adjacency


def test_instantiate_and_score():
    plan = instantiate_template("compact_2bed")
    assert len(plan.rooms) >= 5
    assert plan.doors
    assert plan.windows
    scores = score_plan(plan)
    assert 0 <= scores.overall <= 10
    assert scores.rules
    rules = evaluate_rules(plan)
    assert any(r.hard for r in rules)


def test_adapt_boundary():
    plan = instantiate_template("studio", width=7, depth=5.5)
    boundary = generate_boundary(width=12, depth=9, shape="rect")
    adapted = adapt_to_boundary(plan, boundary)
    assert adapted.boundary.coords()[1][0] == 12


def test_mutations():
    plan = instantiate_template("compact_2bed")
    mirrored, meta = apply_mutation(plan, MutationOp.MIRROR, seed=1)
    assert meta["applied"]
    assert mirrored.id != plan.id
    mutated, history, _ = mutate_pipeline(plan, seed=42, max_attempts=6)
    assert history
    assert mutated.rooms


def test_renderer():
    plan = instantiate_template("studio")
    svg = render_svg(plan, layers=["boundary", "rooms", "labels", "doors", "windows", "topology"])
    assert "<svg" in svg
    assert "polygon" in svg
    out = render_plan(plan)
    assert out["svg"]
    assert out["png"]
    assert out["metadata"]["room_count"] == len(plan.rooms)


def test_importer():
    png = synthesize_bw_fixture()
    plan = import_bw_image(png, meters_per_pixel=0.05)
    assert len(plan.rooms) >= 3
    adj = build_adjacency(plan)
    assert isinstance(adj, dict)


def test_dataset_generator(tmp_path):
    sample = generate_sample("compact_2bed", seed=7, include_renders=True)
    assert sample.floor_plan
    assert sample.scores
    assert sample.svg
    assert sample.metadata["topology_hash"]
    samples = generate_dataset(n=3, seed=3, out_dir=tmp_path, include_renders=False)
    assert len(samples) == 3
    assert (tmp_path / "index.json").exists()
    hashes = {s.metadata["geometry_hash"] for s in samples}
    assert len(hashes) >= 1


def test_engine_facade():
    engine = FloorPlanEngine()
    plan = engine.instantiate("3bed")
    scores = engine.score(plan)
    assert scores.overall >= 0
    rendered = engine.render(plan)
    assert "svg" in rendered
    info = engine.inspect_room(plan, plan.rooms[0].id)
    assert "area" in info
    assert topology_hash(plan)


def test_api_demo():
    from fastapi.testclient import TestClient
    from thefluxtrain.floorplan.api.app import app

    client = TestClient(app)
    r = client.get("/floorplan/health")
    assert r.status_code == 200
    r = client.get("/floorplan/templates")
    assert r.status_code == 200
    assert len(r.json()) >= 5
    r = client.post("/floorplan/demo?template_id=compact_2bed")
    assert r.status_code == 200
    data = r.json()
    assert "plan" in data and "scores" in data and "render" in data
    r = client.post("/floorplan/render", json={"plan": data["plan"]})
    assert r.status_code == 200
    assert "svg" in r.json()
    r = client.post("/floorplan/evaluate", json={"plan": data["plan"]})
    assert r.status_code == 200
    assert "overall" in r.json()
