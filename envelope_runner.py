import argparse
import pandas as pd
from src.data import Data
from src.match import HungarianMatcher, GreedyMatcher, ParallelGreedyMatcher, ParallelHungarianMatcher, ParallelHungarianIouMatcher
from src.metrics import Metrics
from src.plotter import Plotter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth data.")
    parser.add_argument("--det_path", type=str, required=True, help="Path to detection data.")
    parser.add_argument("--bins_path", type=str, required=True, help="Path to bins configuration file.")
    parser.add_argument("--labels_config_path", type=str, required=True, help="Path to labels configuration file.")
    parser.add_argument("--plot_config_path", type=str, required=True, help="Path to plot configuration file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the outputs.")
    parser.add_argument("--matcher_type", type=str, choices=["hungarian", "greedy_score", "greedy_iou", \
                                                             "parallel_greedy", \
                                                             "parallel_hungarian", "parallel_hungarian_iou"], 
                        default="parallel_hungarian", help="Type of matcher to use.")
    return parser.parse_args()

def process_thresholds(data, thresholds, matcher):
    metrics_dict = {}
    for threshold in thresholds:
        # Create a copy of the data for the current threshold
        data_thresh = data.apply_threshold(threshold)

        # Run matching for the thresholded data
        matcher.process(data_thresh)

        # Initialize metrics and evaluate on the thresholded data
        metrics = Metrics()
        metrics.evaluate(data_thresh)

        # Save metrics data for this threshold
        results_at_threshold = extract_metrics_at_threshold(metrics, threshold)

        # Store the results with the appropriate key format
        metrics_dict[f'incl_thresh_{threshold}'] = metrics.get_metrics_data()['incl']
        metrics_dict[f'excl_thresh_{threshold}'] = metrics.get_metrics_data()['excl']

    return metrics_dict

def extract_metrics_at_threshold(metrics, threshold):
    """Extract only the results corresponding to the specific threshold."""
    metrics_data = metrics.get_metrics_data()  # Get the metrics data
    filtered_data = {}

    # Filter the metrics data to only include results at the specified threshold
    for key, df in metrics_data.items():
        filtered_data[key] = df[df['score_threshold'] == threshold].copy()

    return filtered_data

import pandas as pd

def compose_results(metrics_dict):
    """Compose the results from different thresholds to get a combined output for each bin and family."""
    incl_points = []
    excl_points = []

    # Iterate over all threshold keys to find envelope points
    for key, df in metrics_dict.items():
        if 'incl' in key or 'excl' in key:
            try:
                # Extract threshold value from key
                threshold = int(key.split('_')[-1])

                # Get unique families from the current DataFrame
                unique_families = df['category'].unique()

                for family in unique_families:
                    # Filter rows for the current family
                    family_df = df[df['category'] == family]

                    # Get unique bins within this family
                    unique_bins = family_df['bin'].unique()

                    for bin_value in unique_bins:
                        # Filter rows for the current bin
                        bin_df = family_df[family_df['bin'] == bin_value]

                        # Select the point with the corresponding threshold
                        point = bin_df[bin_df['score_threshold'] == threshold]

                        # Ensure the point exists before selecting
                        if not point.empty:
                            point_data = point.iloc[0].copy()
                            point_data['family'] = family  # Add family info
                            point_data['bin'] = bin_value  # Add bin info

                            if 'incl' in key:
                                incl_points.append(point_data)
                            elif 'excl' in key:
                                excl_points.append(point_data)

            except (IndexError, ValueError):
                print(f"Failed to extract or find threshold for key {key}")

    # Convert the collected points into DataFrames for each type
    incl_envelope = pd.DataFrame(incl_points)
    excl_envelope = pd.DataFrame(excl_points)

    return {'incl': incl_envelope, 'excl': excl_envelope}



if __name__ == "__main__":
    args = parse_arguments()

    # Initialize Data
    data = Data(args.gt_path, args.det_path, args.bins_path, args.labels_config_path, args.plot_config_path)

    # Save initial data and configurations
    data.save_configs(args.output_path)
    data.save_dfs(args.output_path, "input")

    # Initialize matcher
    matcher = {
        'hungarian': HungarianMatcher,
        'greedy': GreedyMatcher,
        'parallel_greedy': ParallelGreedyMatcher,
        'parallel_hungarian': ParallelHungarianMatcher,
        'parallel_hungarian_iou': ParallelHungarianIouMatcher
    }[args.matcher_type]()

    # Define thresholds for processing
    thresholds = list(range(0, 100, 5))  # Example threshold values, customize as needed

    # Process metrics for each threshold
    metrics_dict = process_thresholds(data, thresholds, matcher)

    # Compose the final metrics results to create envelopes
    envelope_results = compose_results(metrics_dict)

    # Add the envelope results to the plotter input
    metrics_dict['incl_envelope'] = envelope_results['incl']
    metrics_dict['excl_envelope'] = envelope_results['excl']

    print (envelope_results['incl'][envelope_results['incl']['bin'] == 'all'])

    # Load the envelope results into Metrics and save
    metrics = Metrics()
    metrics.load_metrics_data(envelope_results)
    metrics.save(args.output_path, include_ignores=False)
    metrics.save(args.output_path, include_ignores=True)

    # Generate and save plots using the combined data
    plotter = Plotter(data.plot_config)
    plotter.create_plots({key : metrics_dict[key] for key in metrics_dict.keys() if key[0] == 'i'})  # Use the dictionary with individual thresholds and envelopes
    plotter.save(args.output_path, subfolder = 'plots/incl')
    plotter.create_plots({key : metrics_dict[key] for key in metrics_dict.keys() if key[0] == 'e'})  # Use the dictionary with individual thresholds and envelopes
    plotter.save(args.output_path, subfolder = 'plots/excl')
    print("Plots created and saved.")


