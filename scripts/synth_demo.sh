#!/usr/bin/env bash
set -e
python src/synth/generate_sprites.py --charset assets/charset_demo.txt --fonts assets/fonts --out data/sprites --size 16
python src/synth/compose_paragraphs.py --sprites data/sprites --out data/demo --num_images 50 --canvas 256
