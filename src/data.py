import pandas as pd
import json

class Data:
    def __init__(self, gt_path, det_path, labels_config_path, bins_config_path):
        self.gt_path = gt_path
        self.det_path = det_path
        self.labels_config_path = labels_config_path
        self.bins_config_path = bins_config_path

        # Load the necessary data
        self.load_labels()
        self.load_bins()
        self.gt_df = self.load_gt()
        self.det_df = self.load_det()

        self.det_df = self.det_df

    def load_labels(self):
        with open(self.labels_config_path, 'r') as f:
            self.labels_config = json.load(f)

    def load_bins(self):
        with open(self.bins_config_path, 'r') as f:
            self.bins_config = json.load(f)

    def load_gt(self):
        return pd.read_csv(self.gt_path, sep='\t')

    def load_det(self):
        return pd.read_csv(self.det_path, sep='\t')

    def save(self, gt_path, det_path):
        self.gt_df.to_csv(gt_path, sep='\t', index=False)
        self.det_df.to_csv(det_path, sep='\t', index=False)

    