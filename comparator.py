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

def create_interactive_plots(eval_data, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine families from available data
    families = set()
    for data in eval_data.values():
        for df in data.values():
            if 'category' in df.columns:  # Assuming 'category' represents families
                families.update(df['category'].unique())

    # Create one plot per family
    for family in families:
        fig = make_subplots(rows=1, cols=1)

        # Get unique bins corresponding to the current family
        bins_for_family = set()
        for data in eval_data.values():
            for df in data.values():
                if 'category' in df.columns and family in df['category'].values:
                    family_bins = df[df['category'] == family]['bin'].unique()
                    bins_for_family.update(family_bins)

        all_bins = sorted(bins_for_family, reverse=True)  # Sort bins from largest to smallest
        folders = list(eval_data.keys())

        # Store visibility for all combinations
        visibility_map = []

        # Plot data for each folder and bin
        for folder in folders:
            for ignore_key, df in eval_data[folder].items():
                if 'category' in df.columns and family in df['category'].values:
                    filtered_df = df[df['category'] == family]
                    for selected_bin in all_bins:
                        bin_data = filtered_df[(filtered_df['bin'] == selected_bin) & ((filtered_df['precision'] != 0) | (filtered_df['recall'] != 0))]  # Exclude points where both precision and recall are (0, 0)

                        trace = go.Scatter(
                            x=bin_data['recall'],
                            y=bin_data['precision'],
                            mode='lines+markers',
                            name=f'{folder} - {selected_bin} ({ignore_key})',
                            visible=True  # Initially visible, controlled by buttons
                        )

                        fig.add_trace(trace)
                        visibility_map.append((folder, selected_bin, trace))

        # Prepare button controls for bins
        bin_buttons = []
        for bin_value in all_bins:
            bin_buttons.append(
                dict(
                    label=f"Bin: {bin_value}",
                    method="update",
                    args=[{"visible": [f' - {bin_value} ' in trace.name for trace in fig.data]}]
                )
            )

        # Update layout with independent controls
        fig.update_layout(
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

        # Save the HTML file
        html_output = os.path.join(output_dir, f'{family}_comparison_dashboard.html')
        fig.write_html(html_output, auto_open=False)

if __name__ == "__main__":
    eval_folders = ["output/test1_greedy", "output/test0_greedy"]  # Add your eval folder paths here
    output_dir = "comparison_outputs"
    eval_data = load_metrics_data(eval_folders)
    create_interactive_plots(eval_data, output_dir)





