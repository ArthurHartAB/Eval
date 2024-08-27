import os

class FolderManager:
    def __init__(self, base_output_path):
        self.base_output_path = base_output_path
        self.gt_path = os.path.join(self.base_output_path, "gt.tsv")
        self.det_path = os.path.join(self.base_output_path, "det.tsv")
        self.matched_gt_path = os.path.join(self.base_output_path, "matched_gt.tsv")
        self.matched_det_path = os.path.join(self.base_output_path, "matched_det.tsv")
        self.metrics_paths = os.path.join(self.base_output_path, "metrics.tsv")
        self.plots_output_dir = os.path.join(self.base_output_path, "plots")

    def create_output_folders(self):
        os.makedirs(self.base_output_path, exist_ok=True)
        os.makedirs(self.plots_output_dir, exist_ok=True)