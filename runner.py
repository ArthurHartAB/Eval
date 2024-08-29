import numpy as np
import pandas as pd
from src.data import Data
from src.folder_manager import FolderManager as FM
from src.match import Matcher, ParallelMatcher, GreedyMatcher, ParallelGreedyMatcher
from src.metrics import PRMetrics
import time


if __name__ == '__main__':
    
    # Initialize the FolderManager
    folder_manager = FM("/home/arthur/Desktop/Eval/output/night_0_greedy")
    folder_manager.create_output_folders()
    
    # Load data
    gt_path = "./dfs/night/gt.tsv"
    det_path = "./dfs/night/res_df_ep0.tsv"
    labels_config_path = "./src/configs/labels.json"
    bins_config_path = "./src/configs/bins.json"
    
    data = Data(gt_path, det_path, labels_config_path, bins_config_path)
    
    # Save the initial GT and DET data
    data.save(folder_manager.gt_path, folder_manager.det_path)
    
    # Perform matching
    matcher = ParallelGreedyMatcher(n_jobs=5)
    #matcher = ParallelMatcher(n_jobs=5)  # Use all available cores
    #matcher = Matcher()
    #matcher = GreedyMatcher()

    t = time.time()
    matcher.process_data(data)
    print ("Matching time : ",time.time() - t)
    
    # Save the matched data
    data.save(folder_manager.matched_gt_path, folder_manager.matched_det_path)
    
    # Compute and save metrics
    metrics = PRMetrics()
    metrics.compute(data)
    metrics.save(folder_manager.metrics_paths,include_ignores=True)
    metrics.save(folder_manager.metrics_paths, include_ignores=False)
    
    # Generate and save PR curve plots as HTML files
    metrics.plot_and_save_html(folder_manager.plots_output_dir, include_ignores=True)
    metrics.plot_and_save_html(folder_manager.plots_output_dir, include_ignores=False)



