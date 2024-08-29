import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from src.folder_manager import FolderManager

# Function to load metrics data from multiple folders
def load_metrics_data(eval_folders):
    eval_data = {}
    for folder in eval_folders:
        eval_name = os.path.basename(folder)
        folder_manager = FolderManager(folder)

        # Load metrics files
        included_metrics_path = f"{folder_manager.metrics_paths[:-4]}_incl_ignores.tsv"
        excluded_metrics_path = f"{folder_manager.metrics_paths[:-4]}_excl_ignores.tsv"

        eval_data[eval_name] = {
            'included': pd.read_csv(included_metrics_path, sep='\t'),
            'excluded': pd.read_csv(excluded_metrics_path, sep='\t')
        }
    return eval_data

# Function to extract the maximum value from bin strings and sort bins accordingly
def sort_bins_by_max(bins):
    def extract_max(bin_str):
        max_value, min_value = bin_str.split('-')
        return int(max_value)  # Convert to integer for numeric sorting

    return sorted(bins, key=extract_max, reverse=True)

# Function to darken a color
def darken_color(rgb_color, factor=0.7):
    # Extract RGB values from the 'rgb(r, g, b)' string
    r, g, b = map(int, rgb_color[4:-1].split(', '))
    # Apply the darkening factor
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    # Return the darkened RGB color as a string
    return f'rgb({r}, {g}, {b})'

def create_interactive_plots(eval_data, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine families from available data
    families = set()
    for data in eval_data.values():
        for df in data.values():
            if 'category' in df.columns:  # Assuming 'category' represents families
                families.update(df['category'].unique())

    # Define base colors for different folders
    base_colors = {
        'blue': 'rgb(0, 0, 255)',
        'orange': 'rgb(255, 165, 0)',
        'green': 'rgb(0, 128, 0)',
        'red': 'rgb(255, 0, 0)',
        'purple': 'rgb(128, 0, 128)'
    }  # Base colors for different folders

    # Create one plot per family
    for family in families:
        fig_pr = make_subplots(rows=1, cols=1)
        fig_fa_rand = make_subplots(rows=1, cols=1)

        # Get unique bins corresponding to the current family
        bins_for_family = set()
        for data in eval_data.values():
            for df in data.values():
                if 'category' in df.columns and family in df['category'].values:
                    family_bins = df[df['category'] == family]['bin'].unique()
                    bins_for_family.update(family_bins)

        # Sort bins by maximum value
        all_bins = sort_bins_by_max(bins_for_family)
        folders = list(eval_data.keys())

        # Store visibility for all combinations
        visibility_map_pr = []
        visibility_map_fa_rand = []

        # Plot data for each folder and bin
        for folder_index, folder in enumerate(folders):
            base_color_key = list(base_colors.keys())[folder_index % len(base_colors)]
            base_color = base_colors[base_color_key]
            darker_color = darken_color(base_color)  # Darker version of the base color
            for ignore_key, df in eval_data[folder].items():
                if 'category' in df.columns and family in df['category'].values:
                    filtered_df = df[df['category'] == family]
                    for selected_bin in all_bins:
                        bin_data = filtered_df[(filtered_df['bin'] == selected_bin) & ((filtered_df['precision'] != 0) | (filtered_df['recall'] != 0))]  # Exclude points where both precision and recall are (0, 0)

                        # Set color based on inclusion/exclusion key
                        color = base_color if ignore_key == 'included' else darker_color

                        # Add threshold values to be shown on hover
                        hover_text = [f"Threshold: {t}" for t in bin_data['score_threshold']]

                        # Precision-Recall Plot
                        trace_pr = go.Scatter(
                            x=bin_data['recall'],
                            y=bin_data['precision'],
                            mode='lines+markers',
                            name=f'{folder} - {selected_bin} ({ignore_key})',
                            text=hover_text,  # Show threshold on hover
                            hoverinfo='text+x+y',  # Display threshold with recall and precision
                            visible=True,  # Initially visible, controlled by buttons
                            marker=dict(color=color)  # Set color based on inclusion/exclusion
                        )

                        # fa_rand vs Recall Plot
                        trace_fa_rand = go.Scatter(
                            x=bin_data['recall'],
                            y=bin_data['fa_rand'],
                            mode='lines+markers',
                            name=f'{folder} - {selected_bin} ({ignore_key})',
                            text=hover_text,  # Show threshold on hover
                            hoverinfo='text+x+y',  # Display threshold with recall and fa_rand
                            visible=True,  # Initially visible, controlled by buttons
                            marker=dict(color=color)  # Set color based on inclusion/exclusion
                        )

                        fig_pr.add_trace(trace_pr)
                        fig_fa_rand.add_trace(trace_fa_rand)
                        visibility_map_pr.append((folder, selected_bin, trace_pr))
                        visibility_map_fa_rand.append((folder, selected_bin, trace_fa_rand))

        # Prepare button controls for bins
        bin_buttons = []
        for bin_value in all_bins:
            bin_buttons.append(
                dict(
                    label=f"Bin: {bin_value}",
                    method="update",
                    args=[{"visible": [f' - {bin_value} ' in trace.name for trace in fig_pr.data]}]
                )
            )

        # Update layout with independent controls for PR plot
        fig_pr.update_layout(
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
            title_text=f"Precision-Recall Curves Comparison for {family}"
        )

        # Update layout with independent controls for fa_rand vs recall plot
        fig_fa_rand.update_layout(
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
            title_text=f"fa_rand vs Recall Comparison for {family}"
        )

        # Save the HTML files
        html_output_pr = os.path.join(output_dir, f'{family}_pr_comparison_dashboard.html')
        fig_pr.write_html(html_output_pr, auto_open=False)

        html_output_fa_rand = os.path.join(output_dir, f'{family}_fa_rand_comparison_dashboard.html')
        fig_fa_rand.write_html(html_output_fa_rand, auto_open=False)

if __name__ == "__main__":
    eval_folders = ["output/night_0_greedy", "output/night_0_hungarian"]  # Add your eval folder paths here
    output_dir = "night_comparison_outputs_greedy_hung"
    eval_data = load_metrics_data(eval_folders)
    create_interactive_plots(eval_data, output_dir)











