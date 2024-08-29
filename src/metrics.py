import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import os

class PRMetrics:
    def __init__(self):
        self.results = {'included': {}, 'excluded': {}}

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
            self.results['included'][category] = []
            self.results['excluded'][category] = []

            for min_height, max_height in bin_ranges:
                bin_label = f'{min_height}-{max_height}'

                # Filter GT and DET based on the assigned bins and match status
                gt_filtered = data.gt_df[
                    (data.gt_df['bin'] == bin_label) & 
                    (data.gt_df['label'].isin(data.labels_config[category]['gt_labels']))
                ]
                det_filtered = data.det_df[
                    (data.det_df['bin'] == bin_label) & 
                    (data.det_df['label'].isin(data.labels_config[category]['det_labels']))
                ]

                assert (len(gt_filtered[(gt_filtered.match_status == 'tp') & gt_filtered.eval_ignore]) == \
                        len(det_filtered[(det_filtered.match_status == 'tp') & det_filtered.eval_ignore])), \
                    "eval_ignore tp different between gt and det"

                num_gt_tp = gt_filtered[gt_filtered['match_status'] == 'tp'].shape[0]
                num_gt_md = gt_filtered[gt_filtered['match_status'] == 'md'].shape[0]
                
                num_gt_tp_excl = gt_filtered[
                    (gt_filtered['match_status'] == 'tp') & 
                    (~gt_filtered['eval_ignore'])
                ].shape[0]
                num_gt_md_excl = gt_filtered[
                    (gt_filtered['match_status'] == 'md') & 
                    (~gt_filtered['eval_ignore'])
                ].shape[0]

                score_thresholds = list(range(0, 99, 1))
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

                    # Calculate total missed detections (MD) below the score threshold
                    num_md_below_threshold = num_gt_md + num_gt_tp - num_det_tp_above_score

                    # Precision and recall including ignored matches
                    precision_incl = num_det_tp_above_score / (
                        num_det_tp_above_score + num_det_fa_above_score + num_det_fa_loc_above_score + 
                        num_det_fa_dbl_above_score + num_det_fa_rand_above_score + 1e-9
                    )

                    recall_incl = num_det_tp_above_score / (num_gt_tp + num_gt_md + 1e-9)

                    assert recall_incl <= 1.0, (
                    f"Recall greater than 1: recall_incl={recall_incl}, "
                    f"Category: {category}, Bin: {min_height}-{max_height}, Score threshold: {score}"
                    )

                    self.results['included'][category].append({
                        'bin': bin_label,
                        'score_threshold': score,
                        'precision': precision_incl,
                        'recall': recall_incl,
                        'tp': num_det_tp_above_score,
                        'fa_rand': num_det_fa_rand_above_score,
                        'fa_loc': num_det_fa_loc_above_score,
                        'fa_dbl': num_det_fa_dbl_above_score,
                        'md': num_md_below_threshold
                    })

                    # Excluding matches with ignored GT
                    num_det_tp_above_score_excl = det_filtered[
                        (det_filtered['score'] >= score) & 
                        (det_filtered['match_status'] == 'tp') &
                        (~det_filtered['eval_ignore'])
                    ].shape[0]

                    precision_excl = num_det_tp_above_score_excl / (
                        num_det_tp_above_score_excl + num_det_fa_above_score + num_det_fa_loc_above_score + 
                        num_det_fa_dbl_above_score + num_det_fa_rand_above_score + 1e-9
                    )

                    recall_excl = num_det_tp_above_score_excl / (num_gt_tp_excl + num_gt_md_excl + 1e-9)

                    assert recall_excl <= 1.0, (
                    f"Recall greater than 1: recall_excl={recall_excl}, "
                    f"Category: {category}, Bin: {min_height}-{max_height}, Score threshold: {score}"
                    )

                    self.results['excluded'][category].append({
                        'bin': bin_label,
                        'score_threshold': score,
                        'precision': precision_excl,
                        'recall': recall_excl,
                        'tp': num_det_tp_above_score_excl,
                        'fa_rand': num_det_fa_rand_above_score,
                        'fa_loc': num_det_fa_loc_above_score,
                        'fa_dbl': num_det_fa_dbl_above_score,
                        'md': num_md_below_threshold
                    })


    def save(self, metrics_path, include_ignores=True):
        all_data = []
        key = 'included' if include_ignores else 'excluded'
        for category, metrics in self.results[key].items():
            for metric in metrics:
                all_data.append({
                    'category': category,
                    'bin': metric['bin'],
                    'score_threshold': metric['score_threshold'],
                    'precision': metric['precision'],
                    'recall': metric['recall'],
                    'tp': metric['tp'],
                    'fa_rand': metric['fa_rand'],
                    'fa_loc': metric['fa_loc'],
                    'fa_dbl': metric['fa_dbl'],
                    'md': metric['md']
                })
        
        metrics_df = pd.DataFrame(all_data)
        suffix = '_incl_ignores' if include_ignores else '_excl_ignores'
        metrics_df.to_csv(f"{metrics_path[:-4]}{suffix}.tsv", sep='\t', index=False)


    def plot_and_save_html(self, output_dir, include_ignores=True):
        key = 'included' if include_ignores else 'excluded'
        os.makedirs(output_dir, exist_ok=True)

        for category, metrics in self.results[key].items():
            plots_pr = []
            plots_recall_fa = []
            bins = sorted(list(set([m['bin'] for m in metrics])))

            for bin_range in bins:
                # Filter metrics for this bin
                bin_metrics = [m for m in metrics if m['bin'] == bin_range]

                # Sort by score_threshold for proper plotting
                bin_metrics = sorted(bin_metrics, key=lambda x: x['score_threshold'])

                # Prepare data for PR curve plot, excluding (0, 0) points
                precision = []
                recall = []
                score_thresholds = []
                false_alarms = []

                for m in bin_metrics:
                    if m['precision'] != 0 or m['recall'] != 0:  # Exclude (0, 0)
                        precision.append(m['precision'])
                        recall.append(m['recall'])
                        score_thresholds.append(m['score_threshold'])
                        
                        # Calculate the number of false alarms (false positives)
                        false_alarm_count = (1 - m['precision']) * (m['recall'] * (len(bin_metrics) + 1e-9))
                        false_alarms.append(false_alarm_count)

                # Create PR curve plot for this bin
                trace_pr = go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines+markers',
                    name=f'Bin {bin_range}',
                    text=[f'Score: {s}' for s in score_thresholds]
                )
                plots_pr.append(trace_pr)

                # Create Recall vs FA plot for this bin
                trace_recall_fa = go.Scatter(
                    x=recall,
                    y=false_alarms,
                    mode='lines+markers',
                    name=f'Bin {bin_range}',
                    text=[f'Score: {s}' for s in score_thresholds]
                )
                plots_recall_fa.append(trace_recall_fa)

            # Layout for PR curve plot
            layout_pr = go.Layout(
                title=f'Precision-Recall Curves for {category} ({key})',
                xaxis=dict(title='Recall', range=[0, 1]),
                yaxis=dict(title='Precision', range=[0, 1]),
                hovermode='closest'
            )
            fig_pr = go.Figure(data=plots_pr, layout=layout_pr)

            # Save the PR curve plot as an HTML file
            suffix = '_incl_ignores' if include_ignores else '_excl_ignores'
            html_file_pr = os.path.join(output_dir, f'{category}_pr_curves{suffix}.html')
            pyo.plot(fig_pr, filename=html_file_pr, auto_open=False)

            # Layout for Recall vs FA plot with swapped axes
            layout_recall_fa = go.Layout(
                title=f'Recall vs False Alarms for {category} ({key})',
                xaxis=dict(title='Recall', range=[0, 1]),
                yaxis=dict(title='False Alarms'),
                hovermode='closest'
            )
            fig_recall_fa = go.Figure(data=plots_recall_fa, layout=layout_recall_fa)

            # Save the Recall vs FA plot as an HTML file
            html_file_recall_fa = os.path.join(output_dir, f'{category}_recall_vs_fa{suffix}.html')
            pyo.plot(fig_recall_fa, filename=html_file_recall_fa, auto_open=False)

