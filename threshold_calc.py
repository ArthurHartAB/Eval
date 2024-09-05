import os
import pandas as pd
import plotly.graph_objs as go
import json

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_metrics_data(input_folder):
    """Load the metrics data from the input folder and combine included and excluded."""
    included_metrics_path = os.path.join(input_folder, "metrics_incl_ignores.tsv")
    excluded_metrics_path = os.path.join(input_folder, "metrics_excl_ignores.tsv")
    
    included_metrics = pd.read_csv(included_metrics_path, sep='\t')
    excluded_metrics = pd.read_csv(excluded_metrics_path, sep='\t')

    # Rename fields in included and excluded DataFrames to differentiate them
    rename_dict_incl = {
        'tp': 'tp_incl',
        'md_dbl': 'md_dbl_incl',
        'md_loc': 'md_loc_incl',
        'md_rand': 'md_rand_incl',
        'recall_strict': 'recall_strict_incl',
        'recall_loose': 'recall_loose_incl',
        'precision_strict': 'precision_strict_incl',
        'precision_loose': 'precision_loose_incl'
    }

    rename_dict_excl = {
        'tp': 'tp_excl',
        'md_dbl': 'md_dbl_excl',
        'md_loc': 'md_loc_excl',
        'md_rand': 'md_rand_excl',
        'recall_strict': 'recall_strict_excl',
        'recall_loose': 'recall_loose_excl',
        'precision_strict': 'precision_strict_excl',
        'precision_loose': 'precision_loose_excl'
    }

    included_metrics.rename(columns=rename_dict_incl, inplace=True)
    excluded_metrics.rename(columns=rename_dict_excl, inplace=True)

    # Merge the included and excluded DataFrames on the common columns
    combined_metrics = pd.merge(included_metrics, excluded_metrics, 
                                on=['bin', 'category', 'score_threshold', 'fa_rand', 'fa_loc', 'fa_dbl'], 
                                how='inner')

    return combined_metrics

def evaluate_condition(row, condition_expr):
    """Evaluate the condition using row data."""
    local_vars = {col: row[col] for col in row.index}
    return eval(condition_expr, {}, local_vars)

def sort_bins_by_max(bins):
    """Sort bins by the maximum height value."""
    bins = bins[bins != 'all']
    
    def extract_max(bin_str):
        max_value, min_value = bin_str.split('-')
        return int(max_value)  # Convert to integer for numeric sorting

    return sorted(bins, key=extract_max, reverse=True)

def darken_color(rgb_color, factor=0.1):
    """Darken a given RGB color."""
    r, g, b = map(int, rgb_color[4:-1].split(', '))
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f'rgb({r}, {g}, {b})'

def plot_threshold_analysis(data, config, output_dir):
    """Plot fa_rand, fa_loc, fa_dbl vs score_threshold and draw points for tp_incl and tp_excl conditions."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionary to store threshold data
    threshold_data = {}
    inference_thresholds = {}

    # Define colors for different categories
    base_colors = {
        'fa_dbl': 'rgb(100, 255, 100)',  # Light Green
        'fa_loc': 'rgb(100, 100, 255)',  # Light Orange
        'fa_rand': 'rgb(255, 100, 100)', # Light Red
        'tp_incl': 'rgb(0, 128, 0)',     # Dark Green
        'tp_excl': 'rgb(128, 0, 0)'      # Dark Red
    }

    for category, conditions in config.items():
        ignore_condition = conditions['ignore_condition']
        remove_condition = conditions['remove_condition']

        # Create a figure for the combined plot
        fig_fa_vs_thresh = go.Figure()

        # Prepare lists for visibility controls
        bin_buttons = []
        visibility_map = []

        # Filter data by category
        category_data = data[data['category'] == category]

        # Extract and sort bins
        bins = sort_bins_by_max(category_data['bin'].unique())

        # Initialize threshold data structure for the current category
        if category not in threshold_data:
            threshold_data[category] = {}

        # Prepare data structure for inference thresholds
        inference_thresholds[category] = {
            "bins": [],
            "ignore_threshs": [],
            "remove_threshs": []
        }

        # Plot data for each bin
        for bin_value in bins:
            bin_data = category_data[category_data['bin'] == bin_value]

            score_thresh = bin_data['score_threshold']
            fa_rand = bin_data['fa_rand']
            fa_loc = bin_data['fa_loc']
            fa_dbl = bin_data['fa_dbl']
            tp_incl = bin_data['tp_incl']
            tp_excl = bin_data['tp_excl']

            # Convert bin strings to integer ranges for inference thresholds
            bin_range = list(map(int, bin_value.split('-')))
            inference_thresholds[category]["bins"].append(bin_range)

            # Plot false alarms vs score threshold with defined colors
            trace_fa_rand = go.Scatter(
                x=score_thresh, y=fa_rand,
                mode='lines+markers',
                name=f'Bin {bin_value} fa_rand',
                marker=dict(color=base_colors['fa_rand']),
                visible=(bin_value == bins[0])
            )
            trace_fa_loc = go.Scatter(
                x=score_thresh, y=fa_loc,
                mode='lines+markers',
                name=f'Bin {bin_value} fa_loc',
                marker=dict(color=base_colors['fa_loc']),
                visible=(bin_value == bins[0])
            )
            trace_fa_dbl = go.Scatter(
                x=score_thresh, y=fa_dbl,
                mode='lines+markers',
                name=f'Bin {bin_value} fa_dbl',
                marker=dict(color=base_colors['fa_dbl']),
                visible=(bin_value == bins[0])
            )

            # Add traces for false alarms
            fig_fa_vs_thresh.add_trace(trace_fa_rand)
            fig_fa_vs_thresh.add_trace(trace_fa_loc)
            fig_fa_vs_thresh.add_trace(trace_fa_dbl)

            # Plot true positives for 'included' and 'excluded'
            trace_tp_incl = go.Scatter(
                x=score_thresh, y=tp_incl,
                mode='lines+markers',
                name=f'Bin {bin_value} tp_incl',
                marker=dict(color=base_colors['tp_incl']),
                visible=(bin_value == bins[0])
            )
            trace_tp_excl = go.Scatter(
                x=score_thresh, y=tp_excl,
                mode='lines+markers',
                name=f'Bin {bin_value} tp_excl',
                marker=dict(color=base_colors['tp_excl']),
                visible=(bin_value == bins[0])
            )

            # Add traces for true positives
            fig_fa_vs_thresh.add_trace(trace_tp_incl)
            fig_fa_vs_thresh.add_trace(trace_tp_excl)

            # Determine visibility
            visibility_map.append((bin_value, trace_fa_rand))
            visibility_map.append((bin_value, trace_fa_loc))
            visibility_map.append((bin_value, trace_fa_dbl))
            visibility_map.append((bin_value, trace_tp_incl))
            visibility_map.append((bin_value, trace_tp_excl))

            # Evaluate conditions and find thresholds for each bin
            ignore_threshold = remove_threshold = None
            ignore_min_score = remove_min_score = None
            max_ignore_tp = max_remove_tp = -1  # Start with a minimum tp count

            # Lists to store scores corresponding to the maximum tp
            ignore_scores_for_max_tp = []
            remove_scores_for_max_tp = []

            for i, row in bin_data.iterrows():
                tp_incl_val = row['tp_incl']
                tp_excl_val = row['tp_excl']
                score_val = row['score_threshold']

                # Check for 'ignore' condition and select the maximum tp that satisfies it
                if evaluate_condition(row, ignore_condition):
                    if tp_incl_val > max_ignore_tp:
                        max_ignore_tp = tp_incl_val
                        ignore_threshold = score_val  # Now use score_threshold
                        ignore_scores_for_max_tp = [score_val]
                    elif tp_incl_val == max_ignore_tp:
                        ignore_scores_for_max_tp.append(score_val)

                # Check for 'remove' condition and select the maximum tp that satisfies it
                if evaluate_condition(row, remove_condition):
                    if tp_excl_val > max_remove_tp:
                        max_remove_tp = tp_excl_val
                        remove_threshold = score_val  # Now use score_threshold
                        remove_scores_for_max_tp = [score_val]
                    elif tp_excl_val == max_remove_tp:
                        remove_scores_for_max_tp.append(score_val)

            # Determine the minimum scores for the maximum tp
            if ignore_scores_for_max_tp:
                ignore_min_score = min(ignore_scores_for_max_tp)

            if remove_scores_for_max_tp:
                remove_min_score = min(remove_scores_for_max_tp)

            # Plot thresholds as scatter points for each bin
            if ignore_threshold is not None:
                trace_ignore_thresh = go.Scatter(
                    x=[ignore_threshold, ignore_threshold],
                    y=[0, max(fa_rand.max(), fa_loc.max(), fa_dbl.max(), tp_incl.max(), tp_excl.max())],
                    mode='lines+markers',
                    name=f'Bin {bin_value} ignore threshold score {ignore_min_score}',
                    line=dict(color='Grey'),
                    visible=(bin_value == bins[0])
                )

                fig_fa_vs_thresh.add_trace(trace_ignore_thresh)
                visibility_map.append((bin_value, trace_ignore_thresh))

            if remove_threshold is not None:
                trace_remove_thresh = go.Scatter(
                    x=[remove_threshold, remove_threshold],
                    y=[0, max(fa_rand.max(), fa_loc.max(), fa_dbl.max(), tp_excl.max(), tp_incl.max())],
                    mode='lines+markers',
                    name=f'Bin {bin_value} remove threshold score {remove_min_score}',
                    line=dict(color='Black'),
                    visible=(bin_value == bins[0])
                )

                fig_fa_vs_thresh.add_trace(trace_remove_thresh)
                visibility_map.append((bin_value, trace_remove_thresh))

            # Store threshold data for each bin
            if bin_value not in threshold_data[category]:
                threshold_data[category][bin_value] = {
                    'ignore': None,
                    'remove': None
                }

            if ignore_threshold is not None:
                threshold_data[category][bin_value]['ignore'] = ignore_min_score
                inference_thresholds[category]['ignore_threshs'].append(ignore_min_score)
            else:
                inference_thresholds[category]['ignore_threshs'].append(None)

            if remove_threshold is not None:
                threshold_data[category][bin_value]['remove'] = remove_min_score
                inference_thresholds[category]['remove_threshs'].append(remove_min_score)
            else:
                inference_thresholds[category]['remove_threshs'].append(None)

        # Prepare bin buttons for interactive control
        for bin_value in bins:
            visibility_array = [
                f'{bin_value}' in trace['name']
                for trace in fig_fa_vs_thresh.data
            ]

            bin_buttons.append(
                dict(
                    label=f"Bin: {bin_value}",
                    method="update",
                    args=[{"visible": visibility_array}]
                )
            )

        # Update layout with independent controls
        fig_fa_vs_thresh.update_layout(
            updatemenus=[
                dict(
                    buttons=bin_buttons,
                    direction="down",
                    showactive=True,
                    x=0.3,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ],
            title_text=f"False Alarms and True Positives vs Score Threshold for {category}",
            xaxis=dict(title='Score Threshold'),
            yaxis=dict(title='Counts')
        )

        # Save the HTML files
        html_output = os.path.join(output_dir, f'{category}_fa_vs_thresh_comparison_dashboard.html')
        fig_fa_vs_thresh.write_html(html_output, auto_open=False)

    # Save the thresholds data to JSON files
    family_thresholds_path = os.path.join(output_dir, 'thresholds.json')
    with open(family_thresholds_path, 'w') as json_file:
        json.dump(threshold_data, json_file, indent=4)

    inference_thresholds_path = os.path.join(output_dir, 'inference_thresholds.json')
    with open(inference_thresholds_path, 'w') as json_file:
        json.dump(inference_thresholds, json_file, indent=4)

if __name__ == "__main__":
    # Load configuration
    config_path = "/home/arthur/Desktop/Eval/src/configs/threshold_conditions.json"
    config = load_config(config_path)

    # Set input folder
    input_folder = "/home/arthur/Desktop/Eval/output/default_ep7"

    # Load metrics data
    metrics_data = load_metrics_data(input_folder)

    # Define output directory for plot
    # s
    output_dir = "threshs/default_ep7"

    # Plot and analyze thresholds
    plot_threshold_analysis(metrics_data, config, output_dir)
