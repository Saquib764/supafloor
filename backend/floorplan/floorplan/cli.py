"""CLI entrypoints for local development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import generate_dataset, generate_sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FloorPlan Dataset Engine CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate one sample from scratch")
    p_gen.add_argument("--rooms", type=int, default=3, help="Sleeping room count")
    p_gen.add_argument("--area", type=float, default=None, help="Target encapsulating area (ft²)")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--out", type=str, default=None)
    p_gen.add_argument("--score-level", default=None, choices=["basic", "intermediate", "advance"])
    p_gen.add_argument("--threshold", type=float, default=None, help="Minimum score for --score-level")

    p_ds = sub.add_parser("dataset", help="Generate a dataset")
    p_ds.add_argument("-n", type=int, default=10)
    p_ds.add_argument("--rooms", type=int, default=None, help="Fixed sleeping room count")
    p_ds.add_argument("--seed", type=int, default=0)
    p_ds.add_argument("--out", type=str, default="floorplan_dataset")
    p_ds.add_argument("--renders", action="store_true")
    p_ds.add_argument(
        "--score-level",
        default="basic",
        choices=["basic", "intermediate", "advance"],
        help="Score tier gate (default: basic)",
    )
    p_ds.add_argument("--threshold", type=float, default=None, help="Minimum level score (default: config)")
    p_ds.add_argument("--max-attempts", type=int, default=None, help="Per-slot threshold retries")
    p_ds.add_argument("--workers", type=int, default=1, help="Parallel outer seeds (ordered accept)")
    p_ds.add_argument("--pretty", action="store_true", help="Indent JSON (default: compact)")

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
            score_level=args.score_level,
            score_threshold=args.threshold,
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
                        "bedroom_count": sample.requirements.get("bedroom_count"),
                        "overall": sample.scores.overall if sample.scores else None,
                        "levels": sample.scores.levels if sample.scores else None,
                        "valid": sample.metadata.get("valid"),
                    },
                    indent=2,
                )
            )
        return

    if args.cmd == "dataset":
        samples = generate_dataset(
            n=args.n,
            bedroom_count=args.rooms,
            seed=args.seed,
            out_dir=args.out,
            include_renders=args.renders,
            score_level=args.score_level,
            score_threshold=args.threshold,
            max_attempts=args.max_attempts,
            workers=args.workers,
            compact_json=not args.pretty,
        )
        print(f"Generated {len(samples)} samples → {args.out}")
        return

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
