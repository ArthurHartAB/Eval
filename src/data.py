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


    