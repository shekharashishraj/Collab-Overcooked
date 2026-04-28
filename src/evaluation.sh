#!/bin/bash
cd ~/Collab-Overcooked/src
python_dir=$(which python)

# === Required-collaboration setting (original) ===
/usr/bin/env ${python_dir}  -- evaluation.py --test_mode build_in
/usr/bin/env ${python_dir}  -- organize_result.py
/usr/bin/env ${python_dir}  -- convert_result.py

# === Non-required (open) setting — Level 1 dishes only ===
# Computes structured (constructive) interdependence per episode and aggregates.
# Reads logs from data/open/<model>/<dish>/, writes JSON to eval_result/open/.
mkdir -p eval_result/open
for dish_dir in data/open/*/1_boiled_egg data/open/*/1_boiled_mushroom data/open/*/1_boiled_sweet_potato data/open/*/1_baked_bell_pepper data/open/*/1_baked_sweet_potato data/open/*/boiled_egg data/open/*/boiled_mushroom data/open/*/boiled_sweet_potato data/open/*/baked_bell_pepper data/open/*/baked_sweet_potato; do
  if [ -d "$dish_dir" ]; then
    out_file="eval_result/open/$(echo "$dish_dir" | tr '/' '_')_interdependence.json"
    /usr/bin/env ${python_dir}  -- structured_interdependence.py "$dish_dir" > "$out_file"
    echo "wrote $out_file"
  fi
done
