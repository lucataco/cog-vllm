#!/bin/sh
# Set COG_WEIGHTS in predict.py to a model name or URL
# COG_WEIGHTS = "Qwen/Qwen3-VL-8B-Instruct"
# OR
# COG_WEIGHTS="https://weights.replicate.delivery/default/Qwen/Qwen3-VL-8B-Instruct/model.tar"

exec cog predict -i prompt="Who are you?"
