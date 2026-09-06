# Floorplan VAE (training view)

Hub view over [`saquiboye/floorplan-to-json`](https://huggingface.co/datasets/saquiboye/floorplan-to-json) for image-to-image VAE training.

Loads only the image pair columns:

| Field | Type | Meaning |
|---|---|---|
| `stem` | string | `{plan_id}_{variant}` |
| `plan_id` | string | Unique apartment / plan id |
| `variant` | int | Augmentation index |
| `image` | image | Realistic 1024×1024 floor-plan JPG (input) |
| `simple_image` | image | Simple grayscale render (target): black walls, gray-80 windows, gray-160 doors |

Training streams `image` (input) and `simple_image` (target) from the Hub, both used as 1-channel. It does not materialize an Arrow copy and does not read local export folders.
