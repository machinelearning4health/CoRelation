from main import generate_parser, run, print_metrics
import torch
import numpy as np
def main():
    parser = generate_parser()
    args = parser.parse_args()
    random_seeds = [1234, 2345, 3456, 4567, 5678]
    best_metrics = []
    for round in [1,2,3,4,5]:
        if round < args.round_idx:
            continue
        seed = random_seeds[round-1]
        torch.manual_seed(seed)
        args.round = round
        best_test_metric = run(args)
        best_metrics.append(best_test_metric)
    # average best test metrics
    print("Average best test metrics: ")
    avg_best_test_metric = {}
    std_best_test_metric = {}
    for metric in best_metrics[0].keys():
        avg_best_test_metric[metric] = sum([best_metric[metric] for best_metric in best_metrics])/len(best_metrics)
        std_best_test_metric[metric] = np.std(np.array([best_metric[metric] for best_metric in best_metrics]))
    print_metrics(avg_best_test_metric, 'Test_AVG')
    print_metrics(std_best_test_metric, 'Test_STD')


if __name__ == "__main__":
    main()