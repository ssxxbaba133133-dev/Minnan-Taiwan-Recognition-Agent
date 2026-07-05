# Model weights are not bundled

This remote API package intentionally does not include local vision model weights.

The language model is called through `MODEL_API_BASE_URL`. Image recognition tasks still need the original YOLO/classification files in this folder, or a separate remote vision service.
