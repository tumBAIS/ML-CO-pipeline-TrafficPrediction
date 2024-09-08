#!/bin/bash

PERCENTAGE_NEW="0.1"
SEED_NEW="1"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
PERCENTAGE_ORIGINAL="1"


source venv/bin/activate

cd matsim-berlin/

cd ../surrogate
python calculate_training_data.py --percentage_original=${PERCENTAGE_ORIGINAL} --percentage_new=${PERCENTAGE_NEW} --seed_new=${SEED_NEW} --cutout_seeds ${SEEDS} --simulation="cutoutWorldsWithoutBoundary" --simulator="matsim"

