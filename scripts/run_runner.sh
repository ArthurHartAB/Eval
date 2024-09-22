#!/bin/bash

# Define the paths to input files and configurations
#gt_path="/home/arthur/Data/hard_images/gt_30_4_24.tsv"
gt_path="/home/arthur/Desktop/night_train_tagged/od_night_train_28_11_2023/gt.tsv"
#gt_path="/home/arthur/Desktop/combined_old_new_sets_night_eu_8mp_08_08_2024/annotations/gt.tsv"
labels_config_path="./src/configs/labels_human.json"
bins_config_path="./src/configs/bins.json"
plot_config_path="./src/configs/plot_config.json"
#"/home/arthur/Desktop/Eval/output/loss_weights_ep1"

# Define an array of det_path and corresponding output_path values
declare -a det_paths=(
   # "/home/arthur/Desktop/Eval/dfs/hard/res_df_ep0.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/zero_gamma/res_df_ep1.tsv"
 #   "/home/arthur/Desktop/Eval/dfs/hard/zero_gamma/res_df_ep4.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/zero_gamma/res_df_ep7.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/one_gamma/res_df_ep1.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/one_gamma/res_df_ep4.tsv"
    #"/home/aSrthur/Desktop/Eval/dfs/hard/one_gamma/res_df_ep7.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/one_gamma/res_df_ep12.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/one_gamma/res_df_ep16.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/default/res_df_ep4.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/default/res_df_ep7.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/default/res_df_ep8.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/default_20_peds/res_df_ep1.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/default_20_peds/res_df_ep4.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/default_20_peds/res_df_ep7.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/loss_weights/res_df_ep1.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/loss_weights/res_df_ep4.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/loss_weights/res_df_ep7.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/loss_weights/res_df_ep10.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/loss_weights/res_df_ep13.tsv"

    #"/home/arthur/Desktop/ep_7_low/inputs/res_df_ep7_parsed.tsv"

    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep1.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep2.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep3.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep4.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep5.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep6.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep7.tsv"
    #"/home/arthur/Desktop/Eval/dfs/hard/scratch/res_df_ep8.tsv"

    "/home/arthur/Desktop/night_train_tagged/od_night_train_28_11_2023/detections.tsv"
    #"/home/arthur/Desktop/long_train_no_misc_human_ep4.tsv"
    #"/home/arthur/Desktop/long_train_no_misc_ep4.tsv"
    #"/home/arthur/Desktop/hard_images_latest_results/detections.tsv"

    #"/home/arthur/Desktop/combined_old_new_sets_night_eu_8mp_08_08_2024/output/detections.tsv"
)
declare -a output_paths=(
   #"/home/arthur/Desktop/Eval/output/default_ep0"
    #"/home/arthur/Desktop/Eval/output/zero_gamma_ep1"
  #  "/home/arthur/Desktop/Eval/output/zero_gamma_ep4"
    #"/home/arthur/Desktop/Eval/output/zero_gamma_ep7"
    #"/home/arthur/Desktop/Eval/output/one_gamma_ep1"
    #"/home/arthur/Desktop/Eval/output/one_gamma_ep4"
    #"/home/arthur/Desktop/Eval/output/one_gamma_ep7"
    #"/home/arthur/Desktop/Eval/output/one_gamma_ep12"
    #"/home/arthur/Desktop/Eval/output/one_gamma_ep16"
    #"/home/arthur/Desktop/Eval/output/default_ep4"
    #"/home/arthur/Desktop/Eval/output/default_ep7"
    #"//home/arthur/Desktop/ep_7_low/new_eval_output"
 #   "/home/arthur/Desktop/Eval/output/default_20_peds_ep1"
    #"/home/arthur/Desktop/Eval/output/default_20_peds_ep4"
    #"/home/arthur/Desktop/Eval/output/default_20_peds_ep7"
    #"/home/arthur/Desktop/Eval/output/loss_weights_ep1"
    #"/home/arthur/Desktop/Eval/output/loss_weights_ep4"
    #"/home/arthur/Desktop/Eval/output/loss_weights_ep7"
    #"/home/arthur/Desktop/Eval/output/loss_weights_ep10"
    #"/home/arthur/Desktop/Eval/output/loss_weights_ep13"

    #"/home/arthur/Desktop/Eval/output/scratch_ep1"
    #"/home/arthur/Desktop/Eval/output/scratch_ep2"
    #"/home/arthur/Desktop/Eval/output/scratch_ep3"
    #"/home/arthur/Desktop/Eval/output/scratch_ep4"
    #"/home/arthur/Desktop/Eval/output/scratch_ep5"
    #"/home/arthur/Desktop/Eval/output/scratch_ep6"
    #"/home/arthur/Desktop/Eval/output/scratch_ep7"
    #"/home/arthur/Desktop/Eval/output/scratch_ep8"
   # "/home/arthur/Desktop/Eval/output/default_long_ep7"

   "/home/arthur/Desktop/night_train_tagged/od_night_train_28_11_2023/eval"
   #"/home/arthur/Desktop/Eval/output/long_train_no_misc_human_ep4"
   #"/home/arthur/Desktop/Eval/output/long_train_no_misc_ep4"

    #"/home/arthur/Desktop/combined_old_new_sets_night_eu_8mp_08_08_2024/eval_results"
)

# Loop over the arrays
for i in "${!det_paths[@]}"; do
    det_path="${det_paths[$i]}"
    output_path="${output_paths[$i]}"

    # Run the Python script with the specified arguments
    python3 runner.py --gt_path "$gt_path" \
                      --det_path "$det_path" \
                      --bins_path "$bins_config_path" \
                      --labels_config_path "$labels_config_path" \
                      --plot_config_path "$plot_config_path" \
                      --output_path "$output_path" \
                      --matcher_type "parallel_hungarian"
done
