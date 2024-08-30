import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class Plotter:
    def __init__(self, plot_config):
        self.plot_config = plot_config
        self.figures = []  # To store the generated figures
        self.colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]  # Fixed set of colors

    def create_plots(self, metrics_dict):
        # Create plots for each entry in the metrics dictionary
        for plot_config in self.plot_config['plots']:
            self.create_family_plots(metrics_dict, plot_config)

    def create_family_plots(self, metrics_dict, plot_config):
        # Determine families from available data
        families = set()
        for metrics in metrics_dict.values():
            if 'category' in metrics.columns:  # Assuming 'category' represents families
                families.update(metrics['category'].unique())

        for key in metrics_dict.keys():
            metrics_dict[key] = metrics_dict[key][metrics_dict[key].recall > 0]

        # Create one plot per family
        for family in families:
            traces = []
            bin_visibility = []
            fig = make_subplots(rows=1, cols=1)

            # Prepare data for each family
            for bin_value in sorted(set().union(*[metrics[metrics['category'] == family]['bin'].unique() for metrics in metrics_dict.values()])):
                for index, (name, metrics) in enumerate(metrics_dict.items()):
                    if 'category' in metrics.columns and family in metrics['category'].values:
                        family_data = metrics[metrics['category'] == family]
                        bin_data = family_data[family_data['bin'] == bin_value]
                        if bin_data.empty:
                            continue

                        # Evaluate y-values based on y_metrics expression
                        try:
                            y_values = pd.eval(plot_config['y_metrics'], engine='python', local_dict=bin_data)
                        except Exception as e:
                            print(f"Error evaluating expression '{plot_config['y_metrics']}' for {family}: {e}")
                            continue

                        # Create trace for the plot
                        trace = go.Scatter(
                            x=bin_data[plot_config['x_metric']],
                            y=y_values,
                            mode='lines+markers',
                            name=f'{name} - {bin_value} ({plot_config["y_metrics"]})',
                            marker=dict(color=self.get_color(index)),
                            text=[f'Score: {s}' for s in bin_data['score_threshold']],
                            hoverinfo='text+x+y',
                            visible=(bin_value == sorted(family_data['bin'].unique())[0])  # Initially visible only for the first bin
                        )
                        traces.append(trace)
                        fig.add_trace(trace)

                bin_visibility.append(bin_value)

            # Prepare button controls for bins
            bin_buttons = []
            for bin_value in sorted(set(bin_visibility)):
                bin_buttons.append(
                    dict(
                        label=f"Bin: {bin_value}",
                        method="update",
                        args=[{"visible": [trace.name.endswith(f'{bin_value} ({plot_config["y_metrics"]})') for trace in fig.data]}]
                    )
                )

            # Update layout with independent controls for the family plot
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
                title_text=f"{plot_config['plot_name']} for {family}",
                xaxis=dict(title=plot_config['x_label']),
                yaxis=dict(title=plot_config['y_label'])
            )

            # Store the figure with its configuration for later saving
            self.figures.append((fig, plot_config, family))

    def get_color(self, index):
        """Assign a color to each metric source in a cyclic manner."""
        return self.colors[index % len(self.colors)]  # Cycle through the list of colors

    def save(self, output_dir, subfolder = 'plots'):
        # Save all figures stored in the instance
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)
        for fig, plot_config, family in self.figures:
            # Simplify the file name by using only the family and the plot name
            plot_name = plot_config["plot_name"].replace(" ", "_").lower()
            html_file = os.path.join(output_dir, subfolder, f'{family}_{plot_name}.html')

            # Save the figure to the HTML file
            fig.write_html(html_file, auto_open=False)

