#!/usr/bin/env python3
"""Aggregate image-to-floorplan transform stats into a Markdown summary.

Usage (from backend/floorplan):

    uv run python dev/summarize_dataset_stats.py
    uv run python dev/summarize_dataset_stats.py \\
      --dir ../../image-to-floorplan/basic_5000/stats \\
      --out ../../image-to-floorplan/basic_5000/STATS.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_BOOL = {"true": True, "false": False}
_COUNT_KEYS = (
    "rooms",
    "doors",
    "windows",
    "missing_label_count",
    "missing_door_symbol",
    "pillars",
)


def _load_stats(path: Path) -> dict:
    stats: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        value = raw.strip()
        if value in _BOOL:
            stats[key.strip()] = _BOOL[value]
        elif value.lstrip("-").isdigit():
            stats[key.strip()] = int(value)
        else:
            stats[key.strip()] = value
    return stats


def _mean(total: int, n: int) -> float:
    return total / n if n else 0.0


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{100.0 * part / whole:.1f}%"


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def _hist_table(
    title: str,
    counts: Counter[int],
    whole: int,
    blurb: str,
    *,
    value_header: str,
    count_header: str,
    share_header: str,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        blurb,
        "",
        f"| {value_header} | {count_header} | {share_header} |",
        "|---:|---:|---:|",
    ]
    for value in sorted(counts):
        n = counts[value]
        lines.append(f"| {value} | {n} | {_pct(n, whole)} |")
    lines.append("")
    return lines


def _markdown(stats_dir: Path, rows: list[dict]) -> str:
    n = len(rows)
    plan_rooms: dict[str, int] = {}
    rotated = 0
    missing_label_any = 0
    missing_door_any = 0
    pillar_any = 0
    totals = Counter()
    mins: dict[str, int] = {}
    maxs: dict[str, int] = {}
    missing_label_hist: Counter[int] = Counter()
    missing_door_hist: Counter[int] = Counter()
    pillar_hist: Counter[int] = Counter()

    for stats in rows:
        plan_id = str(stats.get("id") or "")
        rooms = int(stats.get("rooms") or 0)
        if plan_id:
            plan_rooms[plan_id] = rooms
        if stats.get("is_rotated"):
            rotated += 1
        for key in _COUNT_KEYS:
            value = int(stats.get(key) or 0)
            totals[key] += value
            mins[key] = value if key not in mins else min(mins[key], value)
            maxs[key] = value if key not in maxs else max(maxs[key], value)
        missing_labels = int(stats.get("missing_label_count") or 0)
        missing_doors = int(stats.get("missing_door_symbol") or 0)
        pillars = int(stats.get("pillars") or 0)
        missing_label_hist[missing_labels] += 1
        missing_door_hist[missing_doors] += 1
        pillar_hist[pillars] += 1
        if missing_labels:
            missing_label_any += 1
        if missing_doors:
            missing_door_any += 1
        if pillars:
            pillar_any += 1

    n_plans = len(plan_rooms)
    room_hist: Counter[int] = Counter(plan_rooms.values())
    variants_per_plan = _mean(n, n_plans)

    lines = [
        "# What is in this set of floor-plan pictures",
        "",
        "This page is a simple head-count of a large collection of apartment floor-plan drawings. It does **not** say whether a layout is a “good” or “bad” home. It only describes how many apartments we have, how many pictures we drew of them, and how we deliberately changed those pictures (turned them, hid some room names, and so on).",
        "",
        "Think of it like this:",
        "",
        "- An **apartment** is one unique home layout (living room, bedrooms, kitchen, and so on).",
        "- A **picture** is one drawing of that home. We drew **5 pictures of every apartment**, each a little different, the way a photographer might take several shots of the same room from different angles.",
        "",
        "We do that so a computer that later learns to “read” floor plans sees more than one look: not always upright, not always fully labeled, not always with every door swing drawn.",
        "",
        f"Source files for this count: `{stats_dir.resolve()}`",
        "",
        "## How many apartments and pictures do we have?",
        "",
        "This first table is the size of the collection.",
        "",
        "| What we counted | Number | What this means |",
        "|---|---:|---|",
        f"| Apartments | {n_plans} | There are {n_plans:,} different home layouts. |",
        f"| Pictures | {n} | There are {n:,} drawings in total. |",
        f"| Pictures of each apartment | {_fmt(variants_per_plan)} | Every home was drawn {_fmt(variants_per_plan)} times. {n_plans:,} × {_fmt(variants_per_plan)} = {n:,}. |",
        "",
        "## How we changed the pictures",
        "",
        "A real floor-plan scan is messy: it may be rotated, some rooms may have no written name, a door may be only a gap in the wall. We copied those quirks on purpose.",
        "",
        "**Percent of pictures** means: if you lined up 100 drawings, about this many would have that change. The hidden name or missing door swing is only missing from the *drawing*. We still know the true rooms and doors in the answer key the computer is supposed to learn.",
        "",
        "| What we changed | How many pictures | Percent of pictures | What this means |",
        "|---|---:|---:|---|",
        f"| Turned / rotated | {rotated} | {_pct(rotated, n)} | Every drawing was spun to a random angle, so the computer cannot assume “up” on the page is always the front of the home. |",
        f"| At least one room name missing | {missing_label_any} | {_pct(missing_label_any, n)} | About {int(round(100.0 * missing_label_any / n)) if n else 0} out of every 100 drawings leave the name off at least one room (for example the bedroom is still there, but the word “Bedroom” is not printed). |",
        f"| At least one door without a swing | {missing_door_any} | {_pct(missing_door_any, n)} | About {int(round(100.0 * missing_door_any / n)) if n else 0} out of every 100 drawings skip the usual quarter-circle “door swing” on at least one door. The opening in the wall is still there. |",
        f"| At least one extra column | {pillar_any} | {_pct(pillar_any, n)} | About {int(round(100.0 * pillar_any / n)) if n else 0} out of every 100 drawings include a small square “pillar” at a corner. That is decoration on the page, not a new room. |",
        "",
        "The table above asks “does this *picture* have the change at least once?” The next table asks “how often does it happen to each *room* or *door*?”",
        "",
        "| What we counted | Total times it happened | How common | What this means |",
        "|---|---:|---|---|",
        f"| Room names left off | {totals['missing_label_count']} | {_pct(totals['missing_label_count'], totals['rooms'])} of all rooms | Across every drawing, this share of room names was hidden. |",
        f"| Door swings left off | {totals['missing_door_symbol']} | {_pct(totals['missing_door_symbol'], totals['doors'])} of all doors | This share of doors was drawn as a simple gap, with no swing arc. |",
        f"| Extra columns (pillars) | {totals['pillars']} | {_fmt(_mean(totals['pillars'], n))} per picture | On average, a picture has this many pillars — many pictures have none, and some have one or two. |",
        "",
        "## What is typically in one picture?",
        "",
        "For every drawing we counted rooms, doors, windows, missing names, missing door swings, and extra columns. Then we added those counts up and looked at the smallest, largest, and typical picture.",
        "",
        "How to read the numbers:",
        "",
        "- **Added up across all pictures** — if you stacked every drawing and counted that object everywhere, this is the grand total.",
        "- **Smallest picture** — the drawing that had the fewest of that object (a studio might have only 1 room).",
        "- **Largest picture** — the drawing that had the most of that object (a big apartment can have many rooms).",
        "- **Typical picture** — a rough average: “if you picked a drawing at random, about how many would you see?”",
        "",
        "| What we counted | Added up across all pictures | Smallest picture | Largest picture | Typical picture | What this means |",
        "|---|---:|---:|---:|---:|---|",
        f"| Rooms | {totals['rooms']} | {mins.get('rooms', 0)} | {maxs.get('rooms', 0)} | {_fmt(_mean(totals['rooms'], n))} | A typical drawing shows about {_fmt(_mean(totals['rooms'], n))} rooms. |",
        f"| Doors | {totals['doors']} | {mins.get('doors', 0)} | {maxs.get('doors', 0)} | {_fmt(_mean(totals['doors'], n))} | A typical drawing shows about {_fmt(_mean(totals['doors'], n))} doors. |",
        f"| Windows | {totals['windows']} | {mins.get('windows', 0)} | {maxs.get('windows', 0)} | {_fmt(_mean(totals['windows'], n))} | A typical drawing shows about {_fmt(_mean(totals['windows'], n))} windows. |",
        f"| Room names left off | {totals['missing_label_count']} | {mins.get('missing_label_count', 0)} | {maxs.get('missing_label_count', 0)} | {_fmt(_mean(totals['missing_label_count'], n))} | A typical drawing hides about {_fmt(_mean(totals['missing_label_count'], n))} room names. |",
        f"| Door swings left off | {totals['missing_door_symbol']} | {mins.get('missing_door_symbol', 0)} | {maxs.get('missing_door_symbol', 0)} | {_fmt(_mean(totals['missing_door_symbol'], n))} | A typical drawing skips the swing on about {_fmt(_mean(totals['missing_door_symbol'], n))} doors. |",
        f"| Extra columns | {totals['pillars']} | {mins.get('pillars', 0)} | {maxs.get('pillars', 0)} | {_fmt(_mean(totals['pillars'], n))} | A typical drawing has about {_fmt(_mean(totals['pillars'], n))} pillars. |",
        "",
    ]
    lines.extend(
        _hist_table(
            "How big are the apartments?",
            room_hist,
            n_plans,
            "This table is about the **homes**, not the pictures. Drawing the same home several times does not add rooms, so each apartment is counted once. Read a row as: this many homes have this many rooms. The last column is “out of every 100 apartments.”",
            value_header="Number of rooms in the apartment",
            count_header="How many apartments are this size",
            share_header="Out of every 100 apartments",
        )
    )
    lines.extend(
        _hist_table(
            "Pictures with room names left off",
            missing_label_hist,
            n,
            "Imagine each drawing as a labeled map. Sometimes we erase one or more of those labels so the computer has to find the room from the walls alone. Read a row as: this many pictures are missing exactly this many room names.",
            value_header="Room names left off this picture",
            count_header="How many pictures",
            share_header="Out of every 100 pictures",
        )
    )
    lines.extend(
        _hist_table(
            "Pictures with incomplete door drawings",
            missing_door_hist,
            n,
            "On many floor plans a door is shown as a gap in the wall plus a quarter-circle that shows which way the door swings. We sometimes leave off that quarter-circle. Read a row as: this many pictures hide the swing on exactly this many doors.",
            value_header="Doors without a swing in this picture",
            count_header="How many pictures",
            share_header="Out of every 100 pictures",
        )
    )
    lines.extend(
        _hist_table(
            "Pictures with extra columns drawn",
            pillar_hist,
            n,
            "A **pillar** here is a small square drawn on the plan, like a structural column at a corner. It is extra ink on the page. It is not a room, door, or window. Read a row as: this many pictures have exactly this many extra columns.",
            value_header="Extra columns in this picture",
            count_header="How many pictures",
            share_header="Out of every 100 pictures",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    service_root = Path(__file__).resolve().parents[1]
    repo_root = service_root.parents[1]
    default_dir = repo_root / "image-to-floorplan" / "basic_5000" / "stats"
    default_out = repo_root / "image-to-floorplan" / "basic_5000" / "STATS.md"

    parser = argparse.ArgumentParser(
        description="Read transform stats YAML and write a Markdown summary"
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=default_dir,
        help="Folder of per-variant stats YAML (default: ../../image-to-floorplan/basic_5000/stats)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=default_out,
        help="Markdown output path (default: ../../image-to-floorplan/basic_5000/STATS.md)",
    )
    args = parser.parse_args(argv)

    stats_dir = args.dir if args.dir.is_absolute() else service_root / args.dir
    if not stats_dir.is_dir():
        print(f"Not a directory: {stats_dir}", file=sys.stderr)
        return 1

    paths = sorted(stats_dir.glob("*.yml"))
    if not paths:
        print(f"No *.yml stats files in {stats_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, path in enumerate(paths, start=1):
        rows.append(_load_stats(path))
        if i % 5000 == 0 or i == len(paths):
            print(f"read {i}/{len(paths)}", flush=True)

    out = args.out if args.out.is_absolute() else service_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(stats_dir, rows), encoding="utf-8")
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
