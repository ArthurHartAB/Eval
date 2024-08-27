import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import os

class PRMetrics:
    def __init__(self):
        self.results = {}

    def parse_bins(self, bins_config):
        bins = {}
        for category, config in bins_config.items():
            bins[category] = [
                (int(min_height), int(max_height))
                for bin_range in config["bins"]
                for min_height, max_height in [bin_range.split('-')]
            ]
        return bins

    def compute(self, data):
        self.bins = self.parse_bins(data.bins_config)

        for category, bin_ranges in self.bins.items():
            self.results[category] = []
            for min_height, max_height in bin_ranges:
                # Filter GT and DET based on height bins and match status
                gt_filtered = data.gt_df[
                    (data.gt_df['height'] >= min_height) & 
                    (data.gt_df['height'] < max_height) & 
                    (data.gt_df['label'].isin(data.labels_config[category]['gt_labels']))
                ]
                det_filtered = data.det_df[
                    (data.det_df['height'] >= min_height) & 
                    (data.det_df['height'] < max_height) & 
                    (data.det_df['label'].isin(data.labels_config[category]['det_labels']))
                ]

                num_gt_tp = gt_filtered[gt_filtered['match_status'] == 'tp'].shape[0]
                num_gt_md = gt_filtered[gt_filtered['match_status'] == 'md'].shape[0]

                score_thresholds = list(range(0, 99, 5))
                for score in score_thresholds:
                    num_det_tp_above_score = det_filtered[
                        (det_filtered['score'] >= score) & 
                        (det_filtered['match_status'] == 'tp')
                    ].shape[0]
                    
                    num_det_fa_above_score = det_filtered[
                        (det_filtered['score'] >= score) & 
                        (det_filtered['match_status'] == 'fa')
                    ].shape[0]
                    
                    num_det_fa_loc_above_score = det_filtered[
                        (det_filtered['score'] >= score) & 
                        (det_filtered['match_status'] == 'fa_loc')
                    ].shape[0]
                    
                    num_det_fa_dbl_above_score = det_filtered[
                        (det_filtered['score'] >= score) & 
                        (det_filtered['match_status'] == 'fa_dbl')
                    ].shape[0]

                    num_det_fa_rand_above_score = det_filtered[
                        (det_filtered['score'] >= score) & 
                        (det_filtered['match_status'] == 'fa_rand')
                    ].shape[0]

                    precision = num_det_tp_above_score / (
                        num_det_tp_above_score + num_det_fa_above_score + num_det_fa_loc_above_score + 
                        num_det_fa_dbl_above_score + num_det_fa_rand_above_score + 1e-9
                    )
                    
                    recall = num_det_tp_above_score / (num_gt_tp + num_gt_md + 1e-9)

                    self.results[category].append({
                        'bin': f'{min_height}-{max_height}',
                        'score_threshold': score,
                        'precision': precision,
                        'recall': recall
                    })

    def save(self, metrics_path):
        all_data = []
        for category, metrics in self.results.items():
            for metric in metrics:
                all_data.append({
                    'category': category,
                    'bin': metric['bin'],
                    'score_threshold': metric['score_threshold'],
                    'precision': metric['precision'],
                    'recall': metric['recall']
                })
        
        metrics_df = pd.DataFrame(all_data)
        metrics_df.to_csv(metrics_path, sep='\t', index=False)

    def plot_and_save_html(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for category, metrics in self.results.items():
            plots = []
            bins = sorted(list(set([m['bin'] for m in metrics])))
            
            for bin_range in bins:
                # Filter metrics for this bin
                bin_metrics = [m for m in metrics if m['bin'] == bin_range]

                # Sort by score_threshold for proper plotting
                bin_metrics = sorted(bin_metrics, key=lambda x: x['score_threshold'])

                # Create PR curve plot for this bin
                precision = [m['precision'] for m in bin_metrics]
                recall = [m['recall'] for m in bin_metrics]
                score_thresholds = [m['score_threshold'] for m in bin_metrics]

                trace = go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines+markers',
                    name=f'Bin {bin_range}',
                    text=[f'Score: {s}' for s in score_thresholds]
                )
                plots.append(trace)

            layout = go.Layout(
                title=f'Precision-Recall Curves for {category}',
                xaxis=dict(title='Recall', range=[0, 1]),
                yaxis=dict(title='Precision', range=[0, 1]),
                hovermode='closest'
            )
            fig = go.Figure(data=plots, layout=layout)
            
            # Save the plot as an HTML file
            html_file = os.path.join(output_dir, f'{category}_pr_curves.html')
            pyo.plot(fig, filename=html_file, auto_open=False)






