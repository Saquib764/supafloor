# Dev scripts

Manual utilities for Florence trainer test images and loc-token inspection.

Run from the package root (`backend/florence_trainer/`):

```bash
cd backend/florence_trainer

# Fit every image in test/ onto a 1024×1024 white canvas (in place)
uv run python dev/fit_test_canvas.py

# Plot hardcoded <loc_*> pairs onto a 1000×1000 image
uv run python dev/plot_locs.py
```

| Script | Purpose |
|---|---|
| `fit_test_canvas.py` | Scale + center each `test/` image onto a 1024×1024 white square; overwrites sources |
| `plot_locs.py` | Draw hardcoded Florence loc pairs to `dev/output/locs.png` (points, or polylines when `AS_LINE` is true) |

### `plot_locs.py` knobs

Edit globals at the top of the file:

| Variable | Meaning |
|---|---|
| `LOC_TEXT` | Florence generation string (`<loc_*>` pairs, optional `<sep>` segments) |
| `AS_LINE` | `True` → connect each `<sep>` segment as a line; `False` → points only |
| `SIZE` | Output canvas size (default 1000) |
