import pandas as pd
import os.path as osp

class Metrics:
    def __init__(self):
        self.results = {'included': {}, 'excluded': {}}

    def evaluate(self, data):

        for category in data.bins_config.keys():
            self.results['included'][category] = []
            self.results['excluded'][category] = []

            for bin_label in data.bins_config[category]['bins'] + ['all']:

                # Filter data for ground truth (GT) and detections (DET)
                gt_filtered, det_filtered = self.filter_data_by_bin(data, category, bin_label)

                # Check consistency between matched GT and DET
                assert (len(gt_filtered[(gt_filtered.match_status == 'tp') & gt_filtered.eval_ignore]) ==
                        len(det_filtered[(det_filtered.match_status == 'tp') & det_filtered.eval_ignore])), \
                    "eval_ignore tp different between gt and det"

                # Compute counts for both included and excluded cases
                counts_included, total_counts_included = self.compute_counts(gt_filtered, det_filtered, include_ignores=True)
                counts_excluded, total_counts_excluded = self.compute_counts(gt_filtered, det_filtered, include_ignores=False)

                # Compute metrics for both included and excluded cases
                self.compute_metrics_for_bin(counts_included, total_counts_included, category, bin_label, 'included')
                self.compute_metrics_for_bin(counts_excluded, total_counts_excluded, category, bin_label, 'excluded')


    def filter_data_by_bin(self, data, category, bin_label):
        
        if bin_label == 'all':
        # Include all entries except those explicitly labeled as 'out_of_bins'
            gt_bin_condition = ~data.gt_df['bin'].isin(['out_of_bins'])# arthur
            det_bin_condition = ~data.det_df['bin'].isin(['out_of_bins'])
        else:
        # Filter for a specific bin label
            gt_bin_condition = (data.gt_df['bin'] == bin_label)
            det_bin_condition = (data.det_df['bin'] == bin_label)
        
        
        gt_filtered = data.gt_df.loc[
            gt_bin_condition &
            (data.gt_df['label'].isin(data.labels_config[category]['gt_labels']))
        ].copy()

        det_filtered = data.det_df.loc[
            det_bin_condition &
            (data.det_df['label'].isin(data.labels_config[category]['det_labels']))
        ].copy()

        return gt_filtered, det_filtered

    def compute_counts(self, gt_filtered, det_filtered, include_ignores=True):
        score_thresholds = list(range(0, 99, 1))
        counts_per_threshold = []

        if not include_ignores:
            gt_filtered = gt_filtered[~gt_filtered.eval_ignore]
            det_filtered = det_filtered[~det_filtered.eval_ignore]


        # Compute total counts for GT based on inclusion or exclusion of ignored entries
        total_counts = {
            'tp': len(gt_filtered[(gt_filtered['match_status'] == 'tp') ]),
            'md_dbl': len(gt_filtered[(gt_filtered['match_status'] == 'md_dbl')]),
            'md_loc': len(gt_filtered[(gt_filtered['match_status'] == 'md_loc') ]),
            'md_rand': len(gt_filtered[(gt_filtered['match_status'] == 'md_rand')])
        }

        #print ("total_counts : ", total_counts)

        for score in score_thresholds:
            # Compute counts for detections and misses at the given threshold

            det_filtered_score = det_filtered[(det_filtered['score'] >= score)]

            counts = {
                'tp': len(det_filtered_score[(det_filtered_score['match_status'] == 'tp')]),
                'fa_rand': len(det_filtered_score[(det_filtered_score['match_status'] == 'fa_rand')]),
                'fa_loc': len(det_filtered_score[(det_filtered_score['match_status'] == 'fa_loc') ]),
                'fa_dbl': len(det_filtered_score[(det_filtered_score['match_status'] == 'fa_dbl') ]),
                'md_dbl': len(gt_filtered[(gt_filtered['md_dbl_score'] >= score) & (gt_filtered['det_score'] < score) ]),
                'md_loc': len(gt_filtered[(gt_filtered['md_loc_score'] >= score) & (gt_filtered['md_dbl_score'] < score)]),
                'md_rand': len(gt_filtered[(gt_filtered['md_loc_score'] < score) ])
            }

            #print ("counts : ", counts)

            # Calculate remaining missed detections below threshold

            counts_per_threshold.append((score, counts))

        return counts_per_threshold, total_counts

    def compute_metrics_for_bin(self, counts_per_threshold, total_counts, category, bin_label, key):
        for score, counts in counts_per_threshold:
            # Determine the total counts for strict and loose evaluation
            md_strict = counts['md_dbl'] + counts['md_loc'] + counts['md_rand']
            md_loose = counts['md_rand'] + counts['md_loc']
            tp = counts['tp']  # Use the 'tp' from counts

            # Calculate strict precision and recall
            fa_strict = counts['fa_rand'] + counts['fa_loc'] + counts['fa_dbl']
            precision_strict = tp / (tp + fa_strict + 1e-9)
            recall_strict = tp / (tp + md_strict + 1e-9)

            # Calculate loose precision and recall
            total_fa_loose = counts['fa_rand'] + counts['fa_loc']
            precision_loose = tp / (tp + total_fa_loose + 1e-9)
            recall_loose = tp / (tp + md_loose + 1e-9)

            # Ensure the computed recall values are valid
            assert recall_strict <= 1.0, f"Recall greater than 1: recall_strict={recall_strict}, Category: {category}, Bin: {bin_label}, Score threshold: {score}"
            assert recall_loose <= 1.0, f"Recall greater than 1: recall_loose={recall_loose}, Category: {category}, Bin: {bin_label}, Score threshold: {score}"

            # Append results
            self.results[key][category].append({
                'category' : category,
                'bin': bin_label,
                'score_threshold': score,
                'precision_strict': precision_strict,
                'recall_strict': recall_strict,
                'precision_loose': precision_loose,
                'recall_loose': recall_loose,
                'tp': tp,
                'fa_rand': counts['fa_rand'],
                'fa_loc': counts['fa_loc'],
                'fa_dbl': counts['fa_dbl'],
                'md_rand': counts['md_rand'],
                'md_loc': counts['md_loc'],
                'md_dbl': counts['md_dbl']
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
                    'precision_strict': metric['precision_strict'],
                    'recall_strict': metric['recall_strict'],
                    'precision_loose': metric['precision_loose'],
                    'recall_loose': metric['recall_loose'],
                    'tp': metric['tp'],
                    'fa_rand': metric['fa_rand'],
                    'fa_loc': metric['fa_loc'],
                    'fa_dbl': metric['fa_dbl'],
                    'md_rand': metric['md_rand'],
                    'md_loc': metric['md_loc'],
                    'md_dbl': metric['md_dbl']
                })
        
        metrics_df = pd.DataFrame(all_data)
        suffix = '_incl_ignores' if include_ignores else '_excl_ignores'
        save_path = osp.join(output_path, "metrics" + suffix + '.tsv')
        print(save_path)
        metrics_df.to_csv(save_path, sep='\t', index=False)

    def get_metrics_data(self):
        included_data = []
        excluded_data = []

        for category, metrics in self.results['included'].items():
            for metric in metrics:
                included_data.append(metric)

        for category, metrics in self.results['excluded'].items():
            for metric in metrics:
                excluded_data.append(metric)

        included_df = pd.DataFrame(included_data)
        excluded_df = pd.DataFrame(excluded_data)

        return {'incl': included_df, 'excl': excluded_df}
    

    def load_metrics_data(self, data_dict):
        """
        Load the metrics data from a dictionary input.
        The input should be in the same format as the output of get_metrics_data.
        """
        # Extract the included and excluded data from the input dictionary
        included_df = data_dict.get('incl', pd.DataFrame())
        excluded_df = data_dict.get('excl', pd.DataFrame())

        # Reconstruct the included results
        self.results['included'] = self._reconstruct_results_dict(included_df)

        # Reconstruct the excluded results
        self.results['excluded'] = self._reconstruct_results_dict(excluded_df)

    def _reconstruct_results_dict(self, df):
        """
        Helper function to reconstruct the results dictionary from a DataFrame.
        """
        results_dict = {}
        if df.empty:
            return results_dict
        
        for _, row in df.iterrows():
            category = row['category']
            if category not in results_dict:
                results_dict[category] = []
            results_dict[category].append(row.to_dict())
        return results_dict


