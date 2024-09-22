#!/bin/bash

# Define the paths to the evaluation folders (use parentheses and correct array syntax)
EVAL_FOLDERS=(
    #"/home/arthur/Desktop/Eval/output/default_ep0"
    ##"/home/arthur/Desktop/Eval/output/zero_gamma_ep1"
    #"/home/arthur/Desktop/Eval/output/zero_gamma_ep4"
    ##"/home/arthur/Desktop/Eval/output/zero_gamma_ep7"
    #"/home/arthur/Desktop/Eval/output/one_gamma_ep1"
    ##"/home/arthur/Desktop/Eval/output/one_gamma_ep4"
    ##"/home/arthur/Desktop/Eval/output/one_gamma_ep7"
    ##"/home/arthur/Desktop/Eval/output/one_gamma_ep12"
    ##"/home/arthur/Desktop/Eval/output/one_gamma_ep16"
    ##"/home/arthur/Desktop/Eval/output/default_ep4"
    #"/home/arthur/Desktop/Eval/output/default_ep7" # the most stable !!!!
    ##"/home/arthur/Desktop/Eval/output/default_20_peds_ep1"
    ##"/home/arthur/Desktop/Eval/output/default_20_peds_ep4"
    ##"/home/arthur/Desktop/Eval/output/default_20_peds_ep7"
    #"/home/arthur/Desktop/Eval/output/loss_weights_ep1"
    ##"/home/arthur/Desktop/Eval/output/loss_weights_ep4" # the best so far !!!
    ##"/home/arthur/Desktop/Eval/output/loss_weights_ep7"
    ##"/home/arthur/Desktop/Eval/output/loss_weights_ep10"
    ##"/home/arthur/Desktop/Eval/output/loss_weights_ep13"
    #"/home/arthur/Desktop/Eval/output/scratch_ep1"
    #"/home/arthur/Desktop/Eval/output/scratch_ep2"
    #"/home/arthur/Desktop/Eval/output/scratch_ep3"
    #"/home/arthur/Desktop/Eval/output/scratch_ep4"
    #"/home/arthur/Desktop/Eval/output/scratch_ep5"
    #"/home/arthur/Desktop/Eval/output/scratch_ep6"
    #"/home/arthur/Desktop/Eval/output/scratch_ep7"
    #"/home/arthur/Desktop/Eval/output/scratch_ep8"
    #"/home/arthur/Desktop/Eval/output/envelope_hungarian_iou_default_ep7"

    #"/home/arthur/Desktop/Eval/output/default_ep7"
    #"/home/arthur/Desktop/Eval/output/default_ep0"
    #"/home/arthur/Desktop/Eval/output/scratch_ep8"
    #"/home/arthur/Desktop/Eval/output/zero_gamma_ep4"

    #"/home/arthur/git/B2B_eval/output/old_eval_default_ep7_excl"
    #"/home/arthur/git/B2B_eval/output/old_eval_default_ep0_excl"
    #"/home/arthur/git/B2B_eval/output/old_eval_scratch_ep8_excl"
    #"/home/arthur/git/B2B_eval/output/old_eval_zero_gamma_ep4_excl"

    #"/home/arthur/Desktop/Eval/output/default_ep0_ignore_state"
    #"/home/arthur/Desktop/Eval/output/default_ep7_ignore_state"

    #"/home/arthur/Desktop/Eval/output/default_ep7"
    #/home/arthur/Desktop/Eval/output/default_ep8"
    #"/home/arthur/Desktop/Eval/output/default_long_ep7"

    #"/home/arthur/Desktop/Eval/output/no_misc_long_ep1"
    #"/home/arthur/Desktop/Eval/output/no_misc_long_ep4"

    #"/home/arthur/Desktop/Eval/output/envelope_hungarian_default_ep7"
    #"/home/arthur/Desktop/Eval/output/envelope_hungarian_iou_default_ep7"

    #"/home/arthur/Desktop/ep_7_low/old_eval_outputs"
    #"/home/arthur/Desktop/ep_7_low/new_eval_output"

    "/home/arthur/Desktop/Eval/output/long_train_no_misc_ep1"
    "/home/arthur/Desktop/Eval/output/long_train_no_misc_ep4"
    "/home/arthur/Desktop/Eval/output/long_train_no_misc_human_ep1"
    "/home/arthur/Desktop/Eval/output/long_train_no_misc_human_ep4"

    )

# Define the output directory
OUTPUT_DIR="/home/arthur/Desktop/Eval/output/comp_no_misc_human"

# Define the path to the plot configuration file
PLOT_CONFIG="src/configs/plot_config.json"

# Run the Python script with the specified arguments
python3 comparator.py --eval_folders "${EVAL_FOLDERS[@]}" --output_dir "$OUTPUT_DIR" --plot_config "$PLOT_CONFIG"
