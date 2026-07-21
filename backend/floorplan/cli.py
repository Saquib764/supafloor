"""CLI entrypoints for local development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorplan.dataset import generate_dataset, generate_sample
from floorplan.templates import list_templates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FloorPlan Dataset Engine CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("templates", help="List templates")

    p_gen = sub.add_parser("generate", help="Generate one sample")
    p_gen.add_argument("--template", default="compact_2bed")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--out", type=str, default=None)

    p_ds = sub.add_parser("dataset", help="Generate a dataset")
    p_ds.add_argument("-n", type=int, default=10)
    p_ds.add_argument("--seed", type=int, default=0)
    p_ds.add_argument("--out", type=str, default="floorplan_dataset")
    p_ds.add_argument("--renders", action="store_true")

    p_serve = sub.add_parser("serve", help="Run API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.cmd == "templates":
        print(json.dumps(list_templates(), indent=2))
        return

    if args.cmd == "generate":
        sample = generate_sample(args.template, seed=args.seed, include_renders=True)
        payload = sample.model_dump()
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"Wrote {args.out}")
        else:
            print(json.dumps({"id": sample.id, "overall": sample.scores.overall, "valid": sample.metadata.get("valid")}, indent=2))
        return

    if args.cmd == "dataset":
        samples = generate_dataset(n=args.n, seed=args.seed, out_dir=args.out, include_renders=args.renders)
        print(f"Generated {len(samples)} samples → {args.out}")
        return

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("floorplan.api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
