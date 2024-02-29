import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    # input related
    parser.add_argument("--version", type=str, default="mimic3-50",
                        choices=["mimic2", "mimic3", "mimic3-50", "mimic4", "mimic4-50", "mimic4_10", "mimic4_10-50"], help="Dataset version.")
    parser.add_argument("--label_truncate_length", type=int, default=30)
    
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--round", type=int, default=1)
    
    # word encoder
    parser.add_argument("--word", action="store_true")
    parser.add_argument("--word_embedding_path", type=str)
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--word_dim", type=int, default=100)
    parser.add_argument("--word_dp", type=float, default=0.2)
    parser.add_argument("--word_frz", action="store_true")
    
    # combiner
    parser.add_argument("--combiner", type=str, default='lstm',
                    choices=['naive', 'lstm'])
    
    # lstm encoder
    parser.add_argument("--lstm_dp", type=float, default=0.1)
    parser.add_argument("--rnn_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)


    # label related
    parser.add_argument("--label_num_layers", type=int, default=0)
    parser.add_argument("--label_dropout", type=float, default=0.0)
    parser.add_argument("--label_pooling", type=str, default='max',
                        choices=['max', 'mean', 'last'])
    # parser.add_argument("--est_cls", action='store_true')
    parser.add_argument("--est_cls", type=int, default=0)
    
    parser.add_argument("--term_count", type=int, default=1)
    parser.add_argument("--sort_method", type=str, default='random',
                        choices=['max', 'mean', 'random'])

    parser.add_argument("--decoder", type=str, choices=['CoRelationV3', 'CoRelationV4'])
    parser.add_argument("--attention_dim", type=int, default=512)
    parser.add_argument("--attention_head_dim", type=int, default=64)
    parser.add_argument("--without_bias", action="store_true")
    parser.add_argument("--dual_rdrop_alpha", action="store_true")
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--code_embedding_path", type=str)
    parser.add_argument("--rep_dropout", type=float, default=0.2)
    parser.add_argument("--att_dropout", type=float, default=0.0)
    parser.add_argument("--xavier", action="store_true")
    parser.add_argument("--attention_head", type=int, default=1)
    
    parser.add_argument("--head_pooling", type=str, default="max", choices=["max", "concat", "mean", "full"])
    parser.add_argument("--text_pooling", type=str, default="default", choices=["max", "concat", "mean", "full"])
    parser.add_argument("--act_fn_name", type=str, default="tanh", choices=['tanh', 'relu'])
    parser.add_argument("--kl_type", type=str, default="softmax", choices=['softmax', 'sigmoid'])
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--break_loss", action="store_true")
    parser.add_argument("--use_graph_rate", type=float, default=-1.0)
    parser.add_argument("--use_last", action="store_true")
    parser.add_argument("--use_multihead", type=int, default=2)


    # Test setting
    parser.add_argument("--epoch_idx", type=int, default=0)
    parser.add_argument("--round_idx", type=int, default=0)

    # Train setting
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--predict_with", type=str, default='default')
    parser.add_argument("--topk_num", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW", "SGD", "Adam"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_epoch", type=int, default=20)
    parser.add_argument("--early_stop_epoch", type=int, default=-1)
    parser.add_argument("--early_stop_metric", type=str, default="auc_macro")
    # parser.add_argument("--early_stop_strategy", type=str, choices=["best_dev", "last_dev"], default="best_dev")
    parser.add_argument("--truncate_length", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--output_base_dir", type=str, default="./output/")
    parser.add_argument("--prob_threshold", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=['linear', 'constant', 'cosine'])
    parser.add_argument("--random_sub", type=bool, default=False)
    parser.add_argument("--text_block_rate", type=float, default=0.0)

    # loss
    parser.add_argument("--loss_name", type=str, default="ce",
                        choices=['ce', 'focal', 'asy', 'kexue', 'ldam', 'maskce', 'hungarian', 'sigce', 'sigfocal'])
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--focal_alpha", type=float, default=1.0)
    parser.add_argument("--asy_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asy_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asy_clip", type=float, default=0.05)
    parser.add_argument("--able_torch_grad_focal_loss", action="store_true")
    parser.add_argument("--ldam_c", type=float, default=3.0)
    
    parser.add_argument("--rdrop_alpha", type=float, default=0.0)
    parser.add_argument("--with_rdrop_weight", action="store_true")

    parser.add_argument("--main_code_loss_weight", type=float, default=1.0)
    parser.add_argument("--full_weight", type=float, default=1.0)
    parser.add_argument("--code_loss_weight", type=float, default=1.0)
    parser.add_argument("--alpha_weight", type=float, default=0.0)

    
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--threshold_find", type=str, default="v1")

    return parser
