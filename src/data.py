import pandas as pd
import json
import os

class Data:
    def __init__(self, gt_path, det_path, bins_path, labels_config_path, plot_config_path):
        self.gt_path = gt_path
        self.det_path = det_path
        self.bins_path = bins_path
        self.labels_config_path = labels_config_path
        self.plot_config_path = plot_config_path

        # Load data
        self.gt_df = self.load_gt()
        self.det_df = self.load_det()
        self.labels_config = self.load_json(self.labels_config_path)
        self.bins_config = self.load_json(self.bins_path)
        self.plot_config = self.load_json(self.plot_config_path)

        self.bins_config = {key : self.bins_config[key] for key in self.labels_config.keys()}

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def load_gt(self):
        return pd.read_csv(self.gt_path, sep='\t')

    def load_det(self):
        return pd.read_csv(self.det_path, sep='\t')

    def save_dfs(self, output_path, suffix):
        os.makedirs(output_path, exist_ok=True)
        self.gt_df.to_csv(f"{output_path}/gt_{suffix}.tsv", sep='\t', index=False)
        self.det_df.to_csv(f"{output_path}/det_{suffix}.tsv", sep='\t', index=False)

    def save_configs(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/labels_config.json", 'w') as f:
            json.dump(self.labels_config, f)
        with open(f"{output_path}/bins_config.json", 'w') as f:
            json.dump(self.bins_config, f)
        with open(f"{output_path}/plot_config.json", 'w') as f:
            json.dump(self.plot_config, f)

    def apply_threshold(self, threshold):
        """
        Apply a score threshold to filter the detection data.
        Returns a new Data object with the filtered detection data.
        """
        filtered_det_df = self.det_df[self.det_df['score'] >= threshold].copy()

        # Create a new Data object with the filtered detection DataFrame
        filtered_data = Data(
            gt_path=self.gt_path,
            det_path=self.det_path,
            bins_path=self.bins_path,
            labels_config_path=self.labels_config_path,
            plot_config_path=self.plot_config_path
        )

        # Set the filtered detection DataFrame and keep the original ground truth DataFrame
        filtered_data.det_df = filtered_det_df
        filtered_data.gt_df = self.gt_df

        return filtered_data


    