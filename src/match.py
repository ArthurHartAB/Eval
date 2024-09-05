import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

IGNORE_WEIGHT = 0.1
NON_MATCHING_COST = 0
BASE_MATCH_COST = -20
IOU_MATCH_WEIGHT = 0.01
SCORE_MATCH_WEIGHT = 1.0

MATCH_IOU = 0.5
LOC_IOU = 0.3

class HungarianMatcher:
    def process(self, data):
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
        match_result = self.match(cost_matrix, det_scores, gt_filtered['eval_ignore'].values)

        # Add max_iou and match_status columns to the filtered DataFrames
        gt_filtered, det_filtered = self.add_match_columns(gt_filtered, det_filtered, match_result, iou_matrix)

        # Add bin columns based on the bins configuration
        gt_filtered, det_filtered = self.add_bin_column(gt_filtered, det_filtered, bins, match_result)

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
        # Initialize the cost matrix
        cost_matrix = -iou_matrix * IOU_MATCH_WEIGHT + BASE_MATCH_COST  # Initialize cost matrix

        # Penalize low scores by adding a small negative value to the cost
        for j, score in enumerate(scores):
            cost_matrix[:, j] -= score * SCORE_MATCH_WEIGHT  # Smaller penalty for low scores

        # Adjust cost for ignored ground truths
        for i, ignore in enumerate(eval_ignore):
            if ignore:
                # If ground truth is ignored, set IoU threshold to 0.01
                cost_matrix[i, :] *= IGNORE_WEIGHT  # Reduce the matching score by half
            
        
        cost_matrix[iou_matrix < MATCH_IOU] = NON_MATCHING_COST  # High cost for IoU below 0.5

        return cost_matrix

    def match(self, cost_matrix, scores, eval_ignore):
        # Perform the assignment using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
        # Create a list to store the match results
        match_result = []
    
        # Set to keep track of which GTs and Detections have been matched
        matched_gt = set()
        matched_det = set()
    
        # Iterate over the matches
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < NON_MATCHING_COST:
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
        gt_df['det_score'] = -1.0
        gt_df['md_loc_score'] = -1.0
        gt_df['md_dbl_score'] = -1.0
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

                # Ensure that the eval_ignore flag is correctly transferred
                det_df.at[det_df_idx, 'eval_ignore'] = gt_df.at[gt_df_idx, 'eval_ignore']

                # Store the detection score
                gt_df.at[gt_df_idx, 'det_score'] = det_df.at[det_df_idx, 'score']

        # Process all GT bboxes for max IoU and scores for various thresholds
        for gt_i, gt_idx in gt_index_map.items():
            if iou_matrix.shape[1] > 0:  # Ensure there are detection boxes
                max_iou = iou_matrix[gt_i, :].max()
                gt_df.at[gt_idx, 'max_iou'] = max_iou

                # Find scores for IoU conditions
                scores_iou_loc = det_df.loc[iou_matrix[gt_i, :] > LOC_IOU, 'score']
                scores_iou_dbl = det_df.loc[iou_matrix[gt_i, :] > MATCH_IOU, 'score']

                # Get max scores for IoU > 0.3 and IoU > 0.5
                if not scores_iou_loc.empty:
                    gt_df.at[gt_idx, 'md_loc_score'] = scores_iou_loc.max()
                if not scores_iou_dbl.empty:
                    gt_df.at[gt_idx, 'md_dbl_score'] = scores_iou_dbl.max()

            # Update the match status for missed detections
                if gt_df.at[gt_idx, 'match_status'] == 'md':
                    if gt_df.at[gt_idx, 'md_dbl_score'] != -1:
                        gt_df.at[gt_idx, 'match_status'] = 'md_dbl'
                    elif gt_df.at[gt_idx, 'md_loc_score'] != -1:
                        gt_df.at[gt_idx, 'match_status'] = 'md_loc'
                    else:
                        gt_df.at[gt_idx, 'match_status'] = 'md_rand'

        # Determine the type of false alarms for DET that are not matched (fa)
        for det_i, det_idx in det_index_map.items():
            if det_df.at[det_idx, 'match_status'] == 'fa':
                max_iou_with_any_gt = iou_matrix[:, det_i].max()  # Maximum IoU of this detection with any GT

                if max_iou_with_any_gt > MATCH_IOU:
                    det_df.at[det_idx, 'match_status'] = 'fa_dbl'
                elif max_iou_with_any_gt > LOC_IOU:
                    det_df.at[det_idx, 'match_status'] = 'fa_loc'
                else:
                    det_df.at[det_idx, 'match_status'] = 'fa_rand'

        # Ensure the number of true positives with eval_ignore is consistent
        assert (len(gt_df[(gt_df.match_status == 'tp') & gt_df.eval_ignore]) ==
                len(det_df[(det_df.match_status == 'tp') & det_df.eval_ignore])), \
            "eval_ignore tp different between gt and det"

        return gt_df, det_df


    

    def add_bin_column(self, gt_df, det_df, bins, match_result):
        # Add a 'bin' column to gt_df and det_df initialized to None
        gt_df['bin'] = None
        det_df['bin'] = None

        # Create mappings of DataFrame indices to IoU matrix indices
        gt_index_map = {i: idx for i, idx in enumerate(gt_df.index)}
        det_index_map = {i: idx for i, idx in enumerate(det_df.index)}

        # Assign bins to GT boxes based on height
        for gt_i, gt_idx in gt_index_map.items():
            for bin_range in bins:
                min_height, max_height = map(int, bin_range.split('-'))
                if min_height <= gt_df.at[gt_idx, 'height'] < max_height:
                    gt_df.at[gt_idx, 'bin'] = bin_range
                    break

        # For matched DET, assign the same bin as their corresponding GT using match_result
        for gt_idx, det_idx in match_result:
            if gt_idx in gt_index_map and det_idx in det_index_map:
                gt_df_idx = gt_index_map[gt_idx]
                det_df_idx = det_index_map[det_idx]
                
                # Assign the bin of the matched GT to the detection
                det_df.at[det_df_idx, 'bin'] = gt_df.at[gt_df_idx, 'bin']

        # For unmatched DET, calculate the bin based on their height
        for det_i, det_idx in det_index_map.items():
            if det_df.at[det_idx, 'match_status'] != 'tp':  # Only for unmatched detections
                for bin_range in bins:
                    min_height, max_height = map(int, bin_range.split('-'))
                    if min_height <= det_df.at[det_idx, 'height'] < max_height:
                        det_df.at[det_idx, 'bin'] = bin_range
                        break

        # Handle any unmatched bins
        gt_df.loc[gt_df['bin'].isna(), 'bin'] = 'out_of_bins'
        det_df.loc[det_df['bin'].isna(), 'bin'] = 'out_of_bins'

        return gt_df, det_df

class ParallelHungarianMatcher(HungarianMatcher):
    def __init__(self, n_jobs=5):
        super().__init__()
        self.n_jobs = n_jobs

    def process(self, data):
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


class GreedyMatcher(HungarianMatcher):

    def match(self, cost_matrix, scores, eval_ignore):
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
            best_cost = NON_MATCHING_COST

            for gt_idx in range(num_gt):
                if gt_matched[gt_idx]:
                    continue

                cost = cost_matrix[gt_idx, det_idx]
                if cost < best_cost:
                    best_cost = cost
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_cost < NON_MATCHING_COST:
                # Mark the ground truth and detection as matched
                gt_matched[best_gt_idx] = True
                det_matched[det_idx] = True
                match_result.append((best_gt_idx, det_idx))

        return match_result

class ParallelGreedyMatcher(ParallelHungarianMatcher, GreedyMatcher):
    def __init__(self, n_jobs=5):
        # Initialize both parent classes
        ParallelHungarianMatcher.__init__(self, n_jobs=n_jobs)
        GreedyMatcher.__init__(self)


class HungarianIouMatcher(HungarianMatcher):
    def __init__(self):
        HungarianMatcher.__init__(self)
    
    def calculate_cost_matrix(self, iou_matrix, scores, eval_ignore):
        # Initialize the cost matrix
        cost_matrix = -iou_matrix  # Initialize cost matrix

        # Penalize low scores by adding a small negative value to the cost

        # Adjust cost for ignored ground truths
        for i, ignore in enumerate(eval_ignore):
            if ignore:
                cost_matrix[i, :] *= IGNORE_WEIGHT  # Reduce the matching score by half
            cost_matrix[i, iou_matrix[i, :] < MATCH_IOU] = NON_MATCHING_COST  # High cost for IoU below 0.5

        return cost_matrix
    
class TwoStageHungarianMatcher(HungarianMatcher):
    def match(self, cost_matrix, scores, eval_ignore):
        # Split the cost matrix into two stages: non-ignored and ignored
        non_ignored_indices = np.where(~eval_ignore)[0]  # Indices of non-ignored ground truths
        ignored_indices = np.where(eval_ignore)[0]  # Indices of ignored ground truths

        # Separate cost matrix for non-ignored ground truths
        cost_matrix_non_ignored = cost_matrix[non_ignored_indices, :]

        # Perform the first stage of matching with non-ignored ground truths
        row_ind_non_ignored, col_ind_non_ignored = linear_sum_assignment(cost_matrix_non_ignored)

        # Map the non-ignored matches back to the original cost matrix indices
        row_ind = non_ignored_indices[row_ind_non_ignored]
        col_ind = col_ind_non_ignored

        # Initialize the match result with non-ignored matches
        match_result = list(zip(row_ind, col_ind))

        # Filter out matches with costs below NON_MATCHING_COST after the first stage
        valid_match_result = [(r, c) for r, c in match_result if cost_matrix[r, c] < NON_MATCHING_COST]

        # Identify already matched detections to remove them from the second stage
        matched_detections = set(c for _, c in valid_match_result)

        # Filter out already matched detections from the cost matrix for ignored ground truths
        remaining_det_indices = [i for i in range(cost_matrix.shape[1]) if i not in matched_detections]
        cost_matrix_ignored = cost_matrix[ignored_indices, :][:, remaining_det_indices]

        # Perform the second stage of matching with ignored ground truths on remaining matches
        if cost_matrix_ignored.size > 0:  # Ensure there are remaining detections
            row_ind_ignored, col_ind_ignored = linear_sum_assignment(cost_matrix_ignored)

            # Map the ignored matches back to the original cost matrix indices
            row_ind_ignored = ignored_indices[row_ind_ignored]
            col_ind_ignored = [remaining_det_indices[i] for i in col_ind_ignored]

            # Append the ignored matches to the match result
            ignored_match_result = list(zip(row_ind_ignored, col_ind_ignored))

            # Filter out matches with costs below NON_MATCHING_COST after the second stage
            valid_ignored_match_result = [
                (r, c) for r, c in ignored_match_result if cost_matrix[r, c] < NON_MATCHING_COST
            ]

            # Combine the results from both stages
            valid_match_result.extend(valid_ignored_match_result)

        return valid_match_result

    def process(self, data):
        # Use the existing process method from HungarianMatcher
        super().process(data)



class TwoStageParallelHungarianMatcher(ParallelHungarianMatcher, TwoStageHungarianMatcher):
    def __init__(self, n_jobs=5):
        # Initialize both parent classes
        ParallelHungarianMatcher.__init__(self, n_jobs=n_jobs)
        TwoStageHungarianMatcher.__init__(self)


class TwoStageParallelHungarianMatcher(ParallelHungarianMatcher,TwoStageHungarianMatcher):
    def __init__(self, n_jobs=5):
        # Initialize both parent classes
        ParallelHungarianMatcher.__init__(self, n_jobs=n_jobs)
        TwoStageHungarianMatcher.__init__(self)