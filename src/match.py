import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


class Matcher:

    def process_data(self, data):
        # Initialize lists to store the processed DataFrames for all families
        all_gt_dfs = []
        all_det_dfs = []

        # Iterate through each family in labels_config
        for family, config in data.labels_config.items():
            gt_labels = config['gt_labels']
            det_labels = config['det_labels']
            gt_ignore_condition = config.get('gt_ignore_condition', None)

            bins = data.bins_config[family]['bins']
            # Filter data based on the labels for this family
            gt_filtered_family = data.gt_df[data.gt_df['label'].isin(gt_labels)]
            det_filtered_family = data.det_df[data.det_df['label'].isin(det_labels)]

            # Apply ignore condition to ground truth
            if gt_ignore_condition:
                gt_filtered_family['eval_ignore'] = gt_filtered_family.eval(gt_ignore_condition)
            else:
                gt_filtered_family['eval_ignore'] = False

            # Process each image within this family
            for image_name in tqdm(gt_filtered_family['name'].unique(), desc=f'Processing {family}'):
                # Process the image data
                gt_filtered, det_filtered = self.process_single_image(bins, image_name, gt_filtered_family, det_filtered_family)
                
                # Append the processed DataFrames to the lists
                all_gt_dfs.append(gt_filtered)
                all_det_dfs.append(det_filtered)

        # Concatenate all DataFrames back together
        data.gt_df = pd.concat(all_gt_dfs, ignore_index=True)
        data.det_df = pd.concat(all_det_dfs, ignore_index=True)

    def process_single_image(self, bins, image_name, gt_filtered_family, det_filtered_family):
        # Filter the ground truth and detection data for the current image
        gt_filtered = gt_filtered_family[gt_filtered_family['name'] == image_name]
        det_filtered = det_filtered_family[det_filtered_family['name'] == image_name]

        # Extract bounding boxes and scores
        gt_boxes = gt_filtered[['x_center', 'y_center', 'width', 'height']].values
        det_boxes = det_filtered[['x_center', 'y_center', 'width', 'height']].values
        det_scores = det_filtered['score'].values

        # Calculate IoU matrix and cost matrix
        iou_matrix = self.calculate_ious_matrix(gt_boxes, det_boxes)
        cost_matrix = self.calculate_cost_matrix(iou_matrix, det_scores, gt_filtered['eval_ignore'].values)

        # Perform matching
        match_result = self.match(cost_matrix, det_scores)

        # Add max_iou and match_status columns to the filtered DataFrames
        gt_filtered, det_filtered = self.add_match_columns(gt_filtered, det_filtered, match_result, iou_matrix)

        # Add bin columns based on the bins configuration
        gt_filtered, det_filtered = self.add_bin_column(gt_filtered, det_filtered, bins)

        return gt_filtered, det_filtered

    def calculate_ious_matrix(self, gt_boxes, det_boxes):
        iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)))

        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(det_boxes):
                # Calculate the (x, y) coordinates of the intersection rectangle
                xA = max(gt[0] - gt[2] / 2, det[0] - det[2] / 2)
                yA = max(gt[1] - gt[3] / 2, det[1] - det[3] / 2)
                xB = min(gt[0] + gt[2] / 2, det[0] + det[2] / 2)
                yB = min(gt[1] + gt[3] / 2, det[1] + det[3] / 2)

                # Compute the area of intersection rectangle
                interArea = max(0, xB - xA) * max(0, yB - yA)

                # Compute the area of both the prediction and ground truth rectangles
                boxAArea = gt[2] * gt[3]
                boxBArea = det[2] * det[3]

                # Compute the intersection over union by taking the intersection area
                # and dividing it by the sum of prediction + ground-truth areas - intersection area
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
                iou_matrix[i, j] = iou

        return iou_matrix

    def calculate_cost_matrix(self, iou_matrix, scores, eval_ignore):
        cost_matrix = -iou_matrix*0 #np.zeros_like(iou_matrix)

        # Penalize low scores by adding a small negative value to the cost
        for j, score in enumerate(scores):
            cost_matrix[:, j] -= score# Smaller penalty for low scores

        # Adjust cost for ignored ground truths
        for i, ignore in enumerate(eval_ignore):
            if ignore:
                cost_matrix[i, :] *= 0.5  # Reduce the matching score by half

        # Set cost to a very high value (large number) for pairs with IoU < 0.5
        cost_matrix[iou_matrix < 0.5] = 100
        return cost_matrix

    def match(self, cost_matrix, scores):
        # Perform the assignment using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
        # Create a list to store the match results
        match_result = []
    
        # Set to keep track of which GTs and Detections have been matched
        matched_gt = set()
        matched_det = set()
    
        # Iterate over the matches
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0:
                # Ensure that the match is one-to-one
                if r in matched_gt or c in matched_det:
                    raise ValueError(f"Non-unique matching detected: GT {r} or DET {c} is already matched.")
            
                match_result.append((r, c))
                matched_gt.add(r)
                matched_det.add(c)
    
        # Return the one-to-one match result
        return match_result

    def add_match_columns(self, gt_df, det_df, match_result, iou_matrix):
    # Initialize columns
        gt_df['max_iou'] = 0.0
        gt_df['match_status'] = 'md'
        det_df['max_iou'] = 0.0
        det_df['match_status'] = 'fa'
        det_df['eval_ignore'] = False

        # Create mapping of DataFrame indices to IoU matrix indices
        gt_index_map = {i: idx for i, idx in enumerate(gt_df.index)}
        det_index_map = {i: idx for i, idx in enumerate(det_df.index)}

        # Add max IoU and match status for GT and DET based on match_result
        for gt_idx, det_idx in match_result:
            if gt_idx in gt_index_map and det_idx in det_index_map:
                iou = iou_matrix[gt_idx, det_idx]
                gt_df_idx = gt_index_map[gt_idx]
                det_df_idx = det_index_map[det_idx]

                gt_df.at[gt_df_idx, 'max_iou'] = iou
                gt_df.at[gt_df_idx, 'match_status'] = 'tp'
                det_df.at[det_df_idx, 'max_iou'] = iou
                det_df.at[det_df_idx, 'match_status'] = 'tp'
                det_df.at[det_df_idx, 'eval_ignore'] = gt_df.at[gt_df_idx, 'eval_ignore']
            else:
                print(f"Warning: Invalid gt_idx ({gt_idx}) or det_idx ({det_idx}) in match_result")

    # Identify max IoU for unmatched GT (missed detections)
        for gt_i, gt_idx in gt_index_map.items():
            if gt_df.at[gt_idx, 'match_status'] == 'md':
                if iou_matrix.shape[1] > 0:  # Ensure there are detection boxes
                    gt_df.at[gt_idx, 'max_iou'] = iou_matrix[gt_i, :].max()

    # Identify max IoU for unmatched DET (false alarms)
        for det_i, det_idx in det_index_map.items():
            if det_df.at[det_idx, 'match_status'] == 'fa':
                if iou_matrix.shape[0] > 0:  # Ensure there are ground truth boxes
                    max_iou = iou_matrix[:, det_i].max()
                    det_df.at[det_idx, 'max_iou'] = max_iou
                    if max_iou > 0.5:
                        det_df.at[det_idx, 'match_status'] = 'fa_dbl'
                    elif max_iou > 0.3:
                        det_df.at[det_idx, 'match_status'] = 'fa_loc'
                    else:
                        det_df.at[det_idx, 'match_status'] = 'fa_rand'

        assert (len(gt_df[(gt_df.match_status == 'tp') & gt_df.eval_ignore]) == \
                       len(det_df[(det_df.match_status == 'tp') & det_df.eval_ignore])), \
                       "eval_ignore tp different between gt and det"

        return gt_df, det_df
    

    def add_bin_column(self, gt_df, det_df, bins):
    # Add a 'bin' column to gt_df and det_df initialized to None
        gt_df['bin'] = None
        det_df['bin'] = None

    # Assign bins to GT boxes based on height
        for i, row in gt_df.iterrows():
            for bin_range in bins:
                min_height, max_height = map(int, bin_range.split('-'))
                if min_height <= row['height'] < max_height:
                    gt_df.at[i, 'bin'] = bin_range
                    break

    # For matched DET, assign the same bin as their corresponding GT
        for i, row in det_df.iterrows():
            if row['match_status'] == 'tp':
                matched_gt_idx = gt_df.index[gt_df['max_iou'] == row['max_iou']].tolist()
                if matched_gt_idx:
                    det_df.at[i, 'bin'] = gt_df.at[matched_gt_idx[0], 'bin']

    # For unmatched DET, calculate the bin based on their height
        for i, row in det_df.iterrows():
            if row['match_status'] != 'tp':  # Only for unmatched detections
                for bin_range in bins:
                    min_height, max_height = map(int, bin_range.split('-'))
                    if min_height <= row['height'] < max_height:
                        det_df.at[i, 'bin'] = bin_range
                        break

        return gt_df, det_df




class ParallelMatcher(Matcher):
    def __init__(self, n_jobs=-1):
        super().__init__()
        self.n_jobs = n_jobs

    def process_data(self, data):
        # Initialize lists to store the processed DataFrames for all families
        all_gt_dfs = []
        all_det_dfs = []

        # Iterate through each family in labels_config
        for family, config in data.labels_config.items():
            gt_labels = config['gt_labels']
            det_labels = config['det_labels']
            gt_ignore_condition = config.get('gt_ignore_condition', None)

            bins = data.bins_config[family]['bins']
            # Filter data based on the labels for this family
            gt_filtered_family = data.gt_df[data.gt_df['label'].isin(gt_labels)]
            det_filtered_family = data.det_df[data.det_df['label'].isin(det_labels)]

            if gt_ignore_condition:
                gt_filtered_family['eval_ignore'] = gt_filtered_family.eval(gt_ignore_condition)
            else:
                gt_filtered_family['eval_ignore'] = False

            # Prepare arguments for parallel execution
            image_names = gt_filtered_family['name'].unique()
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_single_image)(
                    bins, image_name, 
                    gt_filtered_family[gt_filtered_family['name'] == image_name], 
                    det_filtered_family[det_filtered_family['name'] == image_name]
                ) 
                for image_name in tqdm(image_names, desc=f"Processing {family} in parallel")
            )

            # Concatenate the results back together
            all_gt_dfs.extend([res[0] for res in results])
            all_det_dfs.extend([res[1] for res in results])

        # Combine all family results into the main data
        data.gt_df = pd.concat(all_gt_dfs, ignore_index=True)
        data.det_df = pd.concat(all_det_dfs, ignore_index=True)


class GreedyMatcher(Matcher):

    def match(self, cost_matrix, scores):
        # Initialize arrays to keep track of matched ground truths and detections
        num_gt, num_det = cost_matrix.shape
        gt_matched = np.full(num_gt, False, dtype=bool)
        det_matched = np.full(num_det, False, dtype=bool)

        # Sort detections by score (highest first)
        sorted_det_indices = np.argsort(-scores)

        match_result = []

        # Greedily match detections to ground truths
        for det_idx in sorted_det_indices:
            best_gt_idx = -1
            best_cost = 100

            for gt_idx in range(num_gt):
                if gt_matched[gt_idx]:
                    continue

                cost = cost_matrix[gt_idx, det_idx]
                if cost < best_cost:
                    best_cost = cost
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_cost < 0:
                # Mark the ground truth and detection as matched
                gt_matched[best_gt_idx] = True
                det_matched[det_idx] = True
                match_result.append((best_gt_idx, det_idx))

        return match_result

class ParallelGreedyMatcher(ParallelMatcher, GreedyMatcher):
    def __init__(self, n_jobs=-1):
        # Initialize both parent classes
        ParallelMatcher.__init__(self, n_jobs=n_jobs)
        GreedyMatcher.__init__(self)