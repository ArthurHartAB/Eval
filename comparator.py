import os
import argparse
import pandas as pd
from src.plotter import Plotter  # Import the Plotter class
import json

# Function to load metrics data from multiple folders
def load_metrics_data(eval_folders):
    eval_data = {}
    for folder in eval_folders:
        eval_name = os.path.basename(folder)

        # Load metrics files
        included_metrics_path = os.path.join(folder, "metrics_incl_ignores.tsv")
        excluded_metrics_path = os.path.join(folder, "metrics_excl_ignores.tsv")

        eval_data[eval_name] = {
            'included': pd.read_csv(included_metrics_path, sep='\t'),
            'excluded': pd.read_csv(excluded_metrics_path, sep='\t')
        }
    return eval_data

# Function to load plot configuration from a file
def load_plot_config(config_path):
    with open(config_path, 'r') as f:
        plot_config = json.load(f)
    return plot_config

def create_and_save_plots(eval_data, plot_config, output_dir):
    # Separate data into included and excluded
    included_data = {}
    excluded_data = {}

    for eval_name, data in eval_data.items():
        included_data[eval_name] = data['included']
        excluded_data[eval_name] = data['excluded'][data['excluded'].bin != 'other']

        #included_data[eval_name] = included_data[eval_name][included_data[eval_name].score_threshold >= 10]
        #excluded_data[eval_name] = excluded_data[eval_name][excluded_data[eval_name].score_threshold >= 10]

    # Initialize plotters for "included" and "excluded"
    plotter_incl = Plotter(plot_config)
    plotter_excl = Plotter(plot_config)

    # Create and save plots for "included" data
    plotter_incl.create_plots(included_data)
    plotter_incl.save(output_dir, "incl_ignores")

    # Create and save plots for "excluded" data
    plotter_excl.create_plots(excluded_data)
    plotter_excl.save(output_dir, "excl_ignores")

def parse_arguments():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the evaluation and plot comparisons.")
    parser.add_argument('--eval_folders', nargs='+', required=True, help="Paths to evaluation folders.")
    parser.add_argument('--output_dir', required=True, help="Output directory for the plots.")
    parser.add_argument('--plot_config', required=True, help="Path to the plot configuration JSON file.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load evaluation data
    eval_data = load_metrics_data(args.eval_folders)

    # Load plot configuration
    plot_config = load_plot_config(args.plot_config)

    # Create and save plots
    create_and_save_plots(eval_data, plot_config, args.output_dir)











