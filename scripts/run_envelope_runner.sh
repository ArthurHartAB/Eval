#!/bin/bash

# Define the paths to input files and configurations
gt_path="/home/arthur/Desktop/Eval/dfs/hard/gt_old.tsv"
det_path="/home/arthur/Desktop/Eval/dfs/hard/default/res_df_ep7.tsv"
labels_config_path="./src/configs/labels.json"
bins_config_path="./src/configs/bins.json"
plot_config_path="./src/configs/plot_config.json"
output_path="/home/arthur/Desktop/Eval/output/envelope_hungarian_default_ep7"
matcher_type="parallel_hungarian"

# Run the envelope_runner.py with the provided arguments
echo "Running envelope_runner.py..."

python3 envelope_runner.py \
  --gt_path "$gt_path" \
  --det_path "$det_path" \
  --bins_path "$bins_config_path" \
  --labels_config_path "$labels_config_path" \
  --plot_config_path "$plot_config_path" \
  --output_path "$output_path" \
  --matcher_type "$matcher_type"

echo "Envelope runner completed."
