import pandas as pd
import os.path as osp

class Metrics:
    def __init__(self):
        self.results = {'included': {}, 'excluded': {}}

    def evaluate(self, data):
        self.bins = self.parse_bins(data.bins_config)

        for category, bin_ranges in self.bins.items():
            self.results['included'][category] = []
            self.results['excluded'][category] = []

            for min_height, max_height in bin_ranges:
                bin_label = f'{min_height}-{max_height}'

                # Filter data for ground truth (GT) and detections (DET)
                gt_filtered, det_filtered = self.filter_data_by_bin(data, category, bin_label)
                
                # Check consistency between matched GT and DET
                assert (len(gt_filtered[(gt_filtered.match_status == 'tp') & gt_filtered.eval_ignore]) ==
                        len(det_filtered[(det_filtered.match_status == 'tp') & det_filtered.eval_ignore])), \
                    "eval_ignore tp different between gt and det"

                # Count the ground truths and detections
                gt_counts = self.count_ground_truths(gt_filtered)
                det_counts = self.compute_detection_counts(det_filtered, gt_counts)

                # Compute metrics for both included and excluded cases
                self.compute_metrics_for_bin(det_counts, gt_counts, category, bin_label, 'included')
                self.compute_metrics_for_bin(det_counts, gt_counts, category, bin_label, 'excluded')

    def parse_bins(self, bins_config):
        bins = {}
        for category, config in bins_config.items():
            bins[category] = [
                (int(min_height), int(max_height))
                for bin_range in config["bins"]
                for min_height, max_height in [bin_range.split('-')]
            ]
        return bins

    def filter_data_by_bin(self, data, category, bin_label):
        gt_filtered = data.gt_df[
            (data.gt_df['bin'] == bin_label) &
            (data.gt_df['label'].isin(data.labels_config[category]['gt_labels']))
        ]
        det_filtered = data.det_df[
            (data.det_df['bin'] == bin_label) &
            (data.det_df['label'].isin(data.labels_config[category]['det_labels']))
        ]
        return gt_filtered, det_filtered

    def count_ground_truths(self, gt_filtered):
        gt_counts = {
            'tp': gt_filtered[gt_filtered['match_status'] == 'tp'].shape[0],
            'md': gt_filtered[gt_filtered['match_status'] == 'md'].shape[0],
            'tp_excl': gt_filtered[(gt_filtered['match_status'] == 'tp') & (~gt_filtered['eval_ignore'])].shape[0],
            'md_excl': gt_filtered[(gt_filtered['match_status'] == 'md') & (~gt_filtered['eval_ignore'])].shape[0]
        }
        return gt_counts

    def compute_detection_counts(self, det_filtered, gt_counts):
        score_thresholds = list(range(0, 99, 1))
        det_counts = []

        for score in score_thresholds:
            counts = {
                'tp': det_filtered[(det_filtered['score'] >= score) & (det_filtered['match_status'] == 'tp')].shape[0],
                'fa': det_filtered[(det_filtered['score'] >= score) & (det_filtered['match_status'] == 'fa')].shape[0],
                'fa_loc': det_filtered[(det_filtered['score'] >= score) & (det_filtered['match_status'] == 'fa_loc')].shape[0],
                'fa_dbl': det_filtered[(det_filtered['score'] >= score) & (det_filtered['match_status'] == 'fa_dbl')].shape[0],
                'fa_rand': det_filtered[(det_filtered['score'] >= score) & (det_filtered['match_status'] == 'fa_rand')].shape[0],
                'tp_excl': det_filtered[
                    (det_filtered['score'] >= score) & 
                    (det_filtered['match_status'] == 'tp') &
                    (~det_filtered['eval_ignore'])
                ].shape[0]
            }
            counts['md_below_threshold'] = gt_counts['md'] + gt_counts['tp'] - counts['tp']
            det_counts.append((score, counts))
        
        return det_counts

    def compute_metrics_for_bin(self, det_counts, gt_counts, category, bin_label, key):
        for score, counts in det_counts:
            if key == 'included':
                precision = counts['tp'] / (
                    counts['tp'] + counts['fa'] + counts['fa_loc'] + counts['fa_dbl'] + counts['fa_rand'] + 1e-9
                )
                recall = counts['tp'] / (gt_counts['tp'] + gt_counts['md'] + 1e-9)
            else:
                precision = counts['tp_excl'] / (
                    counts['tp_excl'] + counts['fa'] + counts['fa_loc'] + counts['fa_dbl'] + counts['fa_rand'] + 1e-9
                )
                recall = counts['tp_excl'] / (gt_counts['tp_excl'] + gt_counts['md_excl'] + 1e-9)

            assert recall <= 1.0, f"Recall greater than 1: recall={recall}, Category: {category}, Bin: {bin_label}, Score threshold: {score}"

            self.results[key][category].append({
                'bin': bin_label,
                'score_threshold': score,
                'precision': precision,
                'recall': recall,
                'tp': counts['tp_excl'] if key == 'excluded' else counts['tp'],
                'fa_rand': counts['fa_rand'],
                'fa_loc': counts['fa_loc'],
                'fa_dbl': counts['fa_dbl'],
                'md': counts['md_below_threshold']
            })

    def save(self, output_path, include_ignores=True):
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
        save_path = osp.join(output_path, "metrics" + suffix + '.tsv')
        print (save_path)
        metrics_df.to_csv(save_path, sep='\t', index=False)

    def get_metrics_data(self):
        # Initialize lists to store data for each category and bin configuration
        included_data = []
        excluded_data = []

        # Prepare the included metrics data
        for category, metrics in self.results['included'].items():
            for metric in metrics:
                included_data.append({
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

        # Prepare the excluded metrics data
        for category, metrics in self.results['excluded'].items():
            for metric in metrics:
                excluded_data.append({
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

        # Convert the lists of dictionaries into DataFrames
        included_df = pd.DataFrame(included_data)
        excluded_df = pd.DataFrame(excluded_data)

        return {'incl' : included_df, 'excl' : excluded_df}

