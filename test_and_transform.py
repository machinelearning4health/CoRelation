import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from data_util import MimicFullDataset
from evaluation import all_metrics, print_metrics
from torch.utils.data import DataLoader
from train_parser import generate_parser
from train_utils import generate_output_folder_name, generate_model
from accelerate import DistributedDataParallelKwargs, Accelerator
from main import eval_func


def run(args):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

    output_basename = generate_output_folder_name(args)
    accelerator.print(output_basename)
    output_path = args.output_base_dir
    print("Path exists:", os.path.exists(output_path))
    model_saved = torch.load(output_path)
    torch.save(model_saved.state_dict(), output_path.replace(".pth", "_state_dict.pth"))

    word_embedding_path = args.word_embedding_path
    accelerator.print(f"Use word embedding from {word_embedding_path}")

    from data_util import my_collate_fn
    train_dataset = MimicFullDataset(args.version, "train", word_embedding_path, args.truncate_length,
                                     args.label_truncate_length, args.term_count, args.sort_method)
    dev_dataset = MimicFullDataset(args.version, "dev", word_embedding_path, args.truncate_length)
    test_dataset = MimicFullDataset(args.version, "test", word_embedding_path, args.truncate_length)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False,
                                num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False,
                                 num_workers=2, pin_memory=True)

    model = generate_model(args, train_dataset).to(accelerator.device)
    model.c_input_word = train_dataset.c_input_word.to(accelerator.device)
    model.c_word_mask = train_dataset.c_word_mask.to(accelerator.device)
    model.c_word_sent = train_dataset.c_word_sent.to(accelerator.device)
    model.rank_index = train_dataset.rank_index.to(accelerator.device)
    model.avg_label_num = train_dataset.avg_label_num if args.with_rdrop_weight else None
    if args.use_graph:
        model.mc_input_word = train_dataset.mc_input_word.to(accelerator.device)
        model.mc_word_mask = train_dataset.mc_word_mask.to(accelerator.device)
        model.mc_word_sent = train_dataset.mc_word_sent.to(accelerator.device)

    model.load_state_dict(model_saved.state_dict())
    model.topk_num = args.topk_num

    accelerator.print(model)
    model = model.to(accelerator.device)

    if accelerator.is_local_main_process:
        dev_metric, _, threshold = eval_func(model, dev_dataloader, args.device, None, True, args)
        print_metrics(dev_metric, 'Dev')
        print('Threshold:', threshold)
        test_metric, _, threshold = eval_func(model, test_dataloader, args.device, threshold, True, args)
        print_metrics(test_metric, 'Test')


def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
