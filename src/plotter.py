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
        """
        Create plots for each entry in the metrics dictionary.
        """
        for plot_config in self.plot_config['plots']:
            self.create_family_plots(metrics_dict, plot_config)

    def create_family_plots(self, metrics_dict, plot_config):
        """
        Create plots for each family in the data.
        """
        families = self.get_families(metrics_dict)
        metrics_dict = self.filter_non_zero_recall(metrics_dict)

        for family in families:
            fig = make_subplots(rows=1, cols=1)
            self.add_family_traces(fig, metrics_dict, plot_config, family)
            self.add_bin_buttons(fig, metrics_dict, family, plot_config)
            self.store_figure(fig, plot_config, family)

    def get_families(self, metrics_dict):
        """
        Extract unique families from the metrics data.
        """
        families = set()
        for metrics in metrics_dict.values():
            if 'category' in metrics.columns:  # Assuming 'category' represents families
                families.update(metrics['category'].unique())
        return families

    def filter_non_zero_recall(self, metrics_dict):
        """
        Filter out entries with zero recall for all metrics in the dictionary.
        """
        for key in metrics_dict.keys():
            metrics_dict[key] = metrics_dict[key][metrics_dict[key].recall_strict > 0.05]
        return metrics_dict

    def add_family_traces(self, fig, metrics_dict, plot_config, family):
        """
        Add traces for each family to the figure.
        """
        bins = self.get_sorted_bins(metrics_dict, family)

        for bin_value in bins:
            for index, (name, metrics) in enumerate(metrics_dict.items()):
                if 'category' in metrics.columns and family in metrics['category'].values:
                    self.add_trace(fig, metrics, family, bin_value, plot_config, name, index)

    def get_sorted_bins(self, metrics_dict, family):
        """
        Get sorted bins for a family, with "all" bin first.
        """
        bins = sorted(set().union(*[metrics[metrics['category'] == family]['bin'].unique() for metrics in metrics_dict.values()]),
                      key=lambda b: (b != 'all', int(b.split('-')[1]) if b != 'all' else float('-inf')))
        return bins

    def add_trace(self, fig, metrics, family, bin_value, plot_config, name, index):
        """
        Add a single trace to the figure.
        """
        family_data = metrics[metrics['category'] == family]
        bin_data = family_data[family_data['bin'] == bin_value]
        if bin_data.empty:
            print("empty bin: ", bin_value)
            return

        # Evaluate y-values based on y_metrics expression
        try:
            x_values = pd.eval(plot_config['x_metric'], engine='python', local_dict=bin_data)
            y_values = pd.eval(plot_config['y_metrics'], engine='python', local_dict=bin_data)
        except Exception as e:
            print(f"Error evaluating expressions '{plot_config['x_metric']}' or '{plot_config['y_metrics']}' for {family}: {e}")
            return

        # Create trace for the plot
        trace = go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=f'{name} - {bin_value} ({plot_config["y_metrics"]})',
            marker=dict(color=self.get_color(index)),
            text=[f'Score: {s}' for s in bin_data['score_threshold']],
            hoverinfo='text+x+y',
            visible=(bin_value == 'all')  # Make all curves for the "all" bin visible
        )
        fig.add_trace(trace)

    def add_bin_buttons(self, fig, metrics_dict, family, plot_config):
        """
        Add bin selection buttons to the figure.
        """
        bins = self.get_sorted_bins(metrics_dict, family)
        bin_buttons = []
        for bin_value in bins:
            bin_buttons.append(
                dict(
                    label=f"Bin: {bin_value}",
                    method="update",
                    args=[{"visible": [trace.name.endswith(f'{bin_value} ({plot_config["y_metrics"]})') for trace in fig.data]}]
                )
            )

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

    def store_figure(self, fig, plot_config, family):
        """
        Store the figure configuration for later saving.
        """
        self.figures.append((fig, plot_config, family))

    def get_color(self, index):
        """
        Assign a color to each metric source in a cyclic manner.
        """
        return self.colors[index % len(self.colors)]  # Cycle through the list of colors

    def save(self, output_dir, subfolder='plots'):
        """
        Save all figures stored in the instance.
        """
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)
        for fig, plot_config, family in self.figures:
            save_name = plot_config["file_suffix"].replace(" ", "_").lower()
            html_file = os.path.join(output_dir, subfolder, f'{family}_{save_name}.html')
            fig.write_html(html_file, auto_open=False)



