import argparse
from src.data import Data
from src.match import HungarianMatcher, \
                      GreedyMatcher, \
                      ParallelGreedyMatcher, \
                      ParallelHungarianMatcher
from src.metrics import Metrics
from src.plotter import Plotter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth data.")
    parser.add_argument("--det_path", type=str, required=True, help="Path to detection data.")
    parser.add_argument("--bins_path", type=str, required=True, help="Path to bins configuration file.")
    parser.add_argument("--labels_config_path", type=str, required=True, help="Path to labels configuration file.")
    parser.add_argument("--plot_config_path", type=str, required=True, help="Path to plot configuration file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the outputs.")
    parser.add_argument("--matcher_type", type=str, choices=["hungarian", "greedy_score", "greedy_iou", "parallel_greedy_score", "parallel_greedy_iou", "parallel_hungarian"], 
                        default="hungarian_parallel", help="Type of matcher to use.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Initialize Data
    data = Data(args.gt_path, args.det_path, args.bins_path, args.labels_config_path, args.plot_config_path)

    # Save initial data and configurations
    data.save_configs(args.output_path)
    data.save_dfs(args.output_path, "input")

    # Initialize and perform matching
    matcher = {
        'hungarian': HungarianMatcher,
        'greedy_score': GreedyMatcher,
        'parallel_greedy_iou': ParallelGreedyMatcher,
        'parallel_hungarian': ParallelHungarianMatcher
    }[args.matcher_type]()
    matcher.process(data)

    # Save matched data
    data.save_dfs(args.output_path, "matched")

    # Compute and save metrics
    metrics = Metrics()
    metrics.evaluate(data)
    metrics.save(args.output_path, include_ignores=False)
    metrics.save(args.output_path, include_ignores=True)

    # Generate and save plots using the metrics data
    plotter = Plotter(data.plot_config)
    metrics_dict = metrics.get_metrics_data()  # Get metrics data directly from the Metrics instance
    plotter.create_plots(metrics_dict)
    plotter.save(args.output_path)

