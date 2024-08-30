import os
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import json
from src.folder_manager import FolderManager

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_metrics_data(folder_manager):
    """Load the metrics data from the input folder using FolderManager."""
    included_metrics_path = f"{folder_manager.metrics_paths[:-4]}_incl_ignores.tsv"
    excluded_metrics_path = f"{folder_manager.metrics_paths[:-4]}_excl_ignores.tsv"
    
    included_metrics = pd.read_csv(included_metrics_path, sep='\t')
    excluded_metrics = pd.read_csv(excluded_metrics_path, sep='\t')
    
    return {'included': included_metrics, 'excluded': excluded_metrics}

def evaluate_condition(row, condition_expr):
    """Evaluate the condition using row data."""
    local_vars = {col: row[col] for col in row.index}
    return eval(condition_expr, {}, local_vars)

def sort_bins_by_max(bins):
    """Sort bins by the maximum height value."""
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
    """Plot fa_rnd, fa_loc, fa_dbl vs tp and draw points for conditions."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionary to store threshold data
    threshold_data = {}

    # Define colors for included (light) and excluded (dark) categories
    base_colors = {
        'fa_dbl': 'rgb(100, 255, 100)',  # Light Green
        'fa_loc': 'rgb(100, 100, 255)',  # Light Orange
        'fa_rand': 'rgb(255, 100, 100)'  # Light Red
    }

    colors = {
        'included': base_colors,
        'excluded': {key: darken_color(color, factor=0.8) for key, color in base_colors.items()}
    }

    for category, conditions in config.items():
        incl_condition = conditions['incl_condition']
        excl_condition = conditions['excl_condition']

        # Create a figure for the combined plot
        fig_fa_vs_tp = go.Figure()

        # Prepare lists for visibility controls
        bin_buttons = []
        visibility_map = []

        for key, df in data.items():
            category_data = df[df['category'] == category]

            # Extract and sort bins
            bins = sort_bins_by_max(category_data['bin'].unique())
            folders = list(data.keys())

            # Initialize threshold data structure for the current category
            if category not in threshold_data:
                threshold_data[category] = {}

            # Plot data for each folder and bin
            for bin_value in bins:
                bin_data = category_data[(category_data['bin'] == bin_value)]

                tp = bin_data['tp']
                fa_rand = bin_data['fa_rand']
                fa_loc = bin_data['fa_loc']
                fa_dbl = bin_data['fa_dbl']

                # Plotting false alarms vs true positives with defined colors
                trace_fa_rand = go.Scatter(
                    x=tp, y=fa_rand,
                    mode='lines+markers',
                    name=f'Bin {bin_value} fa_rand ({key})',
                    marker=dict(color=colors[key]['fa_rand']),
                    visible=(bin_value == bins[0])
                )
                trace_fa_loc = go.Scatter(
                    x=tp, y=fa_loc,
                    mode='lines+markers',
                    name=f'Bin {bin_value} fa_loc ({key})',
                    marker=dict(color=colors[key]['fa_loc']),
                    visible=(bin_value == bins[0])
                )
                trace_fa_dbl = go.Scatter(
                    x=tp, y=fa_dbl,
                    mode='lines+markers',
                    name=f'Bin {bin_value} fa_dbl ({key})',
                    marker=dict(color=colors[key]['fa_dbl']),
                    visible=(bin_value == bins[0])
                )

                fig_fa_vs_tp.add_trace(trace_fa_rand)
                fig_fa_vs_tp.add_trace(trace_fa_loc)
                fig_fa_vs_tp.add_trace(trace_fa_dbl)

                # Determine visibility
                visibility_map.append((key, bin_value, trace_fa_rand))
                visibility_map.append((key, bin_value, trace_fa_loc))
                visibility_map.append((key, bin_value, trace_fa_dbl))

                # Evaluate conditions and find thresholds for each bin
                incl_threshold = excl_threshold = None
                incl_min_score = excl_min_score = None
                max_incl_tp = max_excl_tp = -1  # Start with a minimum tp count

                # Lists to store scores corresponding to the maximum tp
                incl_scores_for_max_tp = []
                excl_scores_for_max_tp = []

                for i, row in bin_data.iterrows():
                    tp_val = row['tp']
                    score_val = row['score_threshold']

                    # Check for 'included' condition and select the maximum tp that satisfies it
                    if key == 'included' and evaluate_condition(row, incl_condition):
                        if tp_val > max_incl_tp:
                            max_incl_tp = tp_val
                            incl_threshold = tp_val
                            incl_scores_for_max_tp = [score_val]
                        elif tp_val == max_incl_tp:
                            incl_scores_for_max_tp.append(score_val)

                    # Check for 'excluded' condition and select the maximum tp that satisfies it
                    if key == 'excluded' and evaluate_condition(row, excl_condition):
                        if tp_val > max_excl_tp:
                            max_excl_tp = tp_val
                            excl_threshold = tp_val
                            excl_scores_for_max_tp = [score_val]
                        elif tp_val == max_excl_tp:
                            excl_scores_for_max_tp.append(score_val)

                # Determine the minimum scores for the maximum tp
                if incl_scores_for_max_tp:
                    incl_min_score = min(incl_scores_for_max_tp)

                if excl_scores_for_max_tp:
                    excl_min_score = min(excl_scores_for_max_tp)

                # Plot thresholds as scatter points for each bin
                if incl_threshold is not None:
                    trace_incl_thresh = go.Scatter(
                        x=[incl_threshold, incl_threshold],
                        y=[0, 100],
                        mode='lines+markers',
                        name=f'Bin {bin_value} score {incl_min_score}',
                        line=dict(color='Grey'),
                        visible=(bin_value == bins[0])
                    )

                    fig_fa_vs_tp.add_trace(trace_incl_thresh)
                    visibility_map.append((key, bin_value, trace_incl_thresh))

                if excl_threshold is not None:
                    trace_excl_thresh = go.Scatter(
                        x=[excl_threshold, excl_threshold],
                        y=[0, 100],
                        mode='lines+markers',
                        name=f'Bin {bin_value} score {excl_min_score}',
                        line=dict(color='Black'),
                        visible=(bin_value == bins[0])
                    )

                    fig_fa_vs_tp.add_trace(trace_excl_thresh)
                    visibility_map.append((key, bin_value, trace_excl_thresh))

                # Store threshold data for each bin
                if bin_value not in threshold_data[category]:
                    threshold_data[category][bin_value] = {
                        'included': None,
                        'excluded': None
                    }

                if incl_threshold is not None:
                    threshold_data[category][bin_value]['included'] = incl_min_score

                if excl_threshold is not None:
                    threshold_data[category][bin_value]['excluded'] = excl_min_score

        # Prepare bin buttons for interactive control
        for bin_value in bins:
            visibility_array = [
                f'{bin_value}' in trace['name']
                for trace in fig_fa_vs_tp.data
            ]

            bin_buttons.append(
                dict(
                    label=f"Bin: {bin_value}",
                    method="update",
                    args=[{"visible": visibility_array}]
                )
            )

        # Update layout with independent controls
        fig_fa_vs_tp.update_layout(
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
            title_text=f"False Alarms vs True Positives with Thresholds for {category}"
        )

        # Save the HTML files
        html_output = os.path.join(output_dir, f'{category}_fa_vs_tp_comparison_dashboard.html')
        fig_fa_vs_tp.write_html(html_output, auto_open=False)

        # Save the thresholds data to a JSON file for each category
        json_output_path = os.path.join(output_dir, f'{category}_thresholds.json')
        with open(json_output_path, 'w') as json_file:
            json.dump(threshold_data, json_file, indent=4)


if __name__ == "__main__":
    # Load configuration
    config_path = "/home/arthur/Desktop/Eval/src/configs/threshold_conditions.json"
    config = load_config(config_path)

    # Set input folder and initialize FolderManager
    input_folder = "/home/arthur/Desktop/Eval/output/night_0_greedy"
    folder_manager = FolderManager(input_folder)

    # Load metrics data
    metrics_data = load_metrics_data(folder_manager)

    # Define output directory for plots
    output_dir = "threshold_analysis_output"

    # Plot and analyze thresholds
    plot_threshold_analysis(metrics_data, config, output_dir)
