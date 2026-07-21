"""CLI entrypoints for local development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorplan.dataset import generate_dataset, generate_sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FloorPlan Dataset Engine CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate one sample from scratch")
    p_gen.add_argument("--rooms", type=int, default=3, help="Sleeping room count")
    p_gen.add_argument("--area", type=float, default=None, help="Target encapsulating area (ft²)")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--out", type=str, default=None)

    p_ds = sub.add_parser("dataset", help="Generate a dataset")
    p_ds.add_argument("-n", type=int, default=10)
    p_ds.add_argument("--rooms", type=int, default=None, help="Fixed sleeping room count")
    p_ds.add_argument("--seed", type=int, default=0)
    p_ds.add_argument("--out", type=str, default="floorplan_dataset")
    p_ds.add_argument("--renders", action="store_true")

    p_serve = sub.add_parser("serve", help="Run API server (prefer: uv run uvicorn app.main:app --reload)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        sample = generate_sample(
            args.rooms,
            target_area=args.area,
            seed=args.seed,
            include_renders=True,
        )
        payload = sample.model_dump()
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"Wrote {args.out}")
        else:
            print(
                json.dumps(
                    {
                        "id": sample.id,
                        "total_rooms": sample.requirements.get("total_rooms"),
                        "overall": sample.scores.overall if sample.scores else None,
                        "valid": sample.metadata.get("valid"),
                    },
                    indent=2,
                )
            )
        return

    if args.cmd == "dataset":
        samples = generate_dataset(
            n=args.n,
            total_rooms=args.rooms,
            seed=args.seed,
            out_dir=args.out,
            include_renders=args.renders,
        )
        print(f"Generated {len(samples)} samples → {args.out}")
        return

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
