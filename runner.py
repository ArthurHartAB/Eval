import numpy as np
import pandas as pd
from src.data import Data
from src.folder_manager import FolderManager as FM
from src.match import Matcher, ParallelMatcher
from src.metrics import PRMetrics

if __name__ == '__main__':
    
    # Initialize the FolderManager
    folder_manager = FM("/home/arthur/Desktop/ART_EVAL/output/test1")
    folder_manager.create_output_folders()
    
    # Load data
    gt_path = "/home/arthur/Desktop/ART_EVAL/dfs/gt_old.tsv"
    det_path = "/home/arthur/Desktop/ART_EVAL/dfs/res_df_ep1.tsv"
    labels_config_path = "./src/configs/labels.json"
    bins_config_path = "./src/configs/bins.json"
    
    data = Data(gt_path, det_path, labels_config_path, bins_config_path)
    
    # Save the initial GT and DET data
    data.save(folder_manager.gt_path, folder_manager.det_path)
    
    # Perform matching
    #matcher = Matcher()
    matcher = ParallelMatcher(n_jobs=5)  # Use all available cores

    matcher.process_data(data)
    
    # Save the matched data
    data.save(folder_manager.matched_gt_path, folder_manager.matched_det_path)
    
    # Compute and save metrics
    metrics = PRMetrics()
    metrics.compute(data)
    metrics.save(folder_manager.metrics_paths)
    
    # Generate and save PR curve plots as HTML files
    metrics.plot_and_save_html(folder_manager.plots_output_dir)



