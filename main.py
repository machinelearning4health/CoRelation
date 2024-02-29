import time
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import random
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from data_util import MimicFullDataset
from tqdm import tqdm
import shutil
import json
import ipdb
import sys
import numpy as np
from constant import MIMIC_2_DIR, MIMIC_3_DIR
from evaluation import all_metrics, print_metrics
from torch.utils.data import DataLoader
from train_parser import generate_parser
from train_utils import generate_output_folder_name, generate_model
from find_threshold import find_threshold_micro, find_threshold_micro_v2
from accelerate import DistributedDataParallelKwargs, Accelerator, InitProcessGroupKwargs
import wandb
import dill
import datetime


def run(args):
    config_list = [args.decoder, args.loss_name, str(args.attention_head), str(args.attention_head_dim), str(args.term_count), str(args.alpha_weight),
     str(args.rdrop_alpha), str(args.use_graph), str(args.without_bias), str(args.sample_num), str(args.round)]
    if args.without_bias:
        config_list.append("without_bias")
    if args.dual_rdrop_alpha:
        config_list.append("dual_rdrop_alpha")
    if args.loss_name == 'sigfocal':
        config_list.append(str(args.focal_gamma))
    if args.topk_num != 300:
        config_list.append(str(args.topk_num))
    init_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60 * 60))
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True), init_kwargs]
    #torch.distributed.init_process_group(backend='nccl', init_method='env://',
    #                                     timeout=datetime.timedelta(seconds=60 * 60))
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
    if accelerator.is_local_main_process:
        wandb.init(project=args.version, name='-'.join(config_list))
    output_basename = generate_output_folder_name(args)
    accelerator.print(output_basename)
    output_path = os.path.join(args.output_base_dir, output_basename)

    try:
        if args.epoch_idx > 0:
            print("Load model from {}".format(os.path.join(output_path, f"epoch{args.epoch_idx}.pth")))
        else:
            os.system(f"mkdir -p {output_path}")
            os.mkdir(output_path)
    except BaseException:
        pass
   
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    word_embedding_path = args.word_embedding_path         
    accelerator.print(f"Use word embedding from {word_embedding_path}")

    from data_util import my_collate_fn
    train_dataset = MimicFullDataset(args.version, "train", word_embedding_path, args.truncate_length,
                                     args.label_truncate_length, args.term_count, args.sort_method)
    dev_dataset = MimicFullDataset(args.version, "dev", word_embedding_path, args.truncate_length)
    test_dataset = MimicFullDataset(args.version, "test", word_embedding_path, args.truncate_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=True, num_workers=2, pin_memory=True)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=2, pin_memory=True)

    if args.epoch_idx > 0:
        print("Load model from {}".format(os.path.join(output_path, f"epoch{args.epoch_idx}.pth")))
        model_old = torch.load(os.path.join(output_path, f"epoch{args.epoch_idx}.pth"), map_location='cpu')
        #model = model.to(accelerator.device)
        #accelerator.print(model)
        model = generate_model(args, train_dataset)
        model.load_state_dict(model_old.state_dict())
        model = model.to(accelerator.device)
        accelerator.print(model)
        optimizer, scheduler_step = model.configure_optimizers(train_dataloader)
        optimizer = optimizer[0]
        scheduler_step = scheduler_step[0]
        steps = 0
        if args.use_last:
            last = torch.load(os.path.join(output_path, f"last{args.epoch_idx}.pth"), pickle_module=dill, map_location='cpu')
            ori_opt_state = last["optimizer"].state_dict()
            optimizer.load_state_dict(ori_opt_state)
            scheduler_step.load_state_dict(last["scheduler"].state_dict())
            steps = last["steps"]
            del last
        else:
            last = torch.load(os.path.join(output_path, f"last{args.epoch_idx}.pth"), pickle_module=dill, map_location='cpu')
            ori_opt_state = last["optimizer"].state_dict()
            ori_opt_state["param_groups"][0]["lr"] = args.learning_rate
            ori_opt_state["param_groups"][1]["lr"] = args.learning_rate
            optimizer.load_state_dict(ori_opt_state)
            #scheduler_step.load_state_dict(last["scheduler"].state_dict())
            #steps = last["steps"]
            del last
        torch.cuda.empty_cache()
    else:
        model = generate_model(args, train_dataset).to(accelerator.device)
        accelerator.print(model)
        optimizer, scheduler_step = model.configure_optimizers(train_dataloader)
        optimizer = optimizer[0]
        scheduler_step = scheduler_step[0]
        steps = 0
    
    # prepare label input feature
    model.c_input_word = train_dataset.c_input_word.to(accelerator.device)
    model.c_word_mask = train_dataset.c_word_mask.to(accelerator.device)
    model.c_word_sent = train_dataset.c_word_sent.to(accelerator.device)
    model.rank_index = train_dataset.rank_index.to(accelerator.device)
    model.avg_label_num = train_dataset.avg_label_num if args.with_rdrop_weight else None
    if args.use_graph:
        model.mc_input_word = train_dataset.mc_input_word.to(accelerator.device)
        model.mc_word_mask = train_dataset.mc_word_mask.to(accelerator.device)
        model.mc_word_sent = train_dataset.mc_word_sent.to(accelerator.device)
    model, optimizer, train_dataloader, scheduler_step = \
        accelerator.prepare(model, optimizer, train_dataloader, scheduler_step)
    if accelerator.is_local_main_process:
        wandb.watch(model, log_freq=200)
    best_dev_metric = {}
    best_test_metric = {}
    early_stop_count = 0
    best_epoch_idx = 0
   
    if accelerator.is_local_main_process and args.debug:
        dev_metric, _, threshold = eval_func(model, dev_dataloader, args.device, None, True, args)
        dev_metric, _, threshold = eval_func(model, test_dataloader, args.device, threshold, True, args)
        print_metrics(dev_metric, 'DEBUG')
    accelerator.print(optimizer)
    # print the state of optimizer
    for var_name in optimizer.state_dict():
        accelerator.print(var_name, "\t", optimizer.state_dict()[var_name])
    accelerator.print(steps)
    s_epoch = args.epoch_idx + 1 if args.use_last else 1
    for epoch_idx in range(s_epoch, args.train_epoch + 1):
        if args.epoch_idx > 0 and epoch_idx <= args.epoch_idx:
            steps = train_one_epoch_dummy(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, args, accelerator)
            accelerator.print(steps)
            accelerator.print(optimizer)
        else:
            epoch_dev_metric, epoch_test_metric, steps = train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, args, accelerator)
        if epoch_idx > args.epoch_idx:
            if accelerator.is_local_main_process:
                # torch.save(models, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
                accelerator.save(accelerator.unwrap_model(model), os.path.join(output_path, f"epoch{epoch_idx}.pth"))
                accelerator.save(accelerator.unwrap_model(model), os.path.join(output_path, f"epoch_last.pth"))
                last = {'optimizer': optimizer, 'scheduler': scheduler_step, 'steps': steps}
                torch.save(last, os.path.join(output_path, f"last{args.epoch_idx}.pth"), pickle_module=dill)
                print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx))
                print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
                print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx))
                print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))

            # Early Stop
            if not best_dev_metric:
                best_dev_metric = epoch_dev_metric
                best_test_metric = epoch_test_metric
                best_epoch_idx = epoch_idx
            else:
                if args.early_stop_metric in epoch_dev_metric:
                    if epoch_dev_metric[args.early_stop_metric] >= best_dev_metric[args.early_stop_metric]:
                        best_dev_metric = epoch_dev_metric
                        best_test_metric = epoch_test_metric
                        best_epoch_idx = epoch_idx
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if args.early_stop_epoch > 0 and early_stop_count >= args.early_stop_epoch:
                accelerator.print(f"Early Stop at Epoch {epoch_idx}, \
                        metric {args.early_stop_metric} not improve on dev set for {early_stop_count} epoch.")
                break
        accelerator.wait_for_everyone()
        
    if accelerator.is_local_main_process:
        best_train_metric, _, _ = eval_func(model, train_dataloader, accelerator.device, args.prob_threshold, True, args)
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx))
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx))
        print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        best_path = os.path.join(output_path, f"epoch{best_epoch_idx}.pth")
        new_path = os.path.join(output_path, "best_epoch.pth")
        # del other epoch except best epoch for mimic3-50
        if args.version == "mimic3-50":
            for del_idx in range(1, args.train_epoch + 1):
                if del_idx != best_epoch_idx:
                    os.system(f'rm {os.path.join(output_path, f"epoch{del_idx}.pth")}')
        os.system(f'cp {best_path} {new_path}')
    if accelerator.is_local_main_process:
        wandb.finish()
    return best_test_metric

def train_one_epoch_dummy(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, accelerator=None):

    epoch_iterator = tqdm(range(len(train_dataloader)), desc="Iteration", ascii=True, disable=not accelerator.is_local_main_process)
    for batch_idx, batch in enumerate(epoch_iterator):

        if scheduler is not None and not args.use_last and args.scheduler == 'linear':
            scheduler.step()  # Update learning rate schedule
            steps += 1
    return steps


def train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, accelerator=None):
    model.train()
    epoch_loss = 0.
    # epoch_mc_loss = 0.
    epoch_kl_loss = 0.
    epoch_c_loss = 0.
    epoch_alpha_loss = 0.

    last_c_loss = []
    last_loss = []
    last_kl_loss = []
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True, disable=not accelerator.is_local_main_process)
    for batch_idx, batch in enumerate(epoch_iterator):
        batch_gpu = tuple([x.to(accelerator.device) for x in batch])
        #if args.rdrop_alpha > 0.0:
        #    ori_loss = models.forward_rdrop(batch_gpu)
        #else:
        ori_loss = model(batch_gpu, rdrop=args.rdrop_alpha > 0.0)
        if isinstance(ori_loss, dict):
            loss = ori_loss['loss']
        else:
            loss = ori_loss

        batch_loss = float(loss.item())
        epoch_loss += batch_loss

        # batch_mc_loss = float(ori_loss['mc_loss'].item())
        # epoch_mc_loss += batch_mc_loss
        batch_c_loss = float(ori_loss['c_loss'].item())
        epoch_c_loss += batch_c_loss
        batch_alpha_loss = float(ori_loss['alpha_loss'].item())
        epoch_alpha_loss += batch_alpha_loss

        if args.rdrop_alpha > 0.0:
            batch_kl_loss = float(ori_loss['kl_loss'].item())
            epoch_kl_loss += batch_kl_loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # loss.backward()

        accelerator.backward(loss)

        if ori_loss['indices_next'] is not None:
            ori_loss = model(batch_gpu, rdrop=args.rdrop_alpha > 0.0, indices=ori_loss['indices_next'])
            if isinstance(ori_loss, dict):
                loss = ori_loss['loss']
            else:
                loss = ori_loss

            batch_loss_fine = float(loss.item())
            batch_loss += batch_loss_fine
            epoch_loss += batch_loss_fine

            # batch_mc_loss = float(ori_loss['mc_loss'].item())
            # epoch_mc_loss += batch_mc_loss
            batch_c_loss_fine = float(ori_loss['c_loss'].item())
            batch_c_loss += batch_c_loss_fine
            epoch_c_loss += batch_c_loss_fine

            batch_alpha_loss_fine = float(ori_loss['alpha_loss'].item())
            batch_alpha_loss += batch_alpha_loss_fine
            epoch_alpha_loss += batch_alpha_loss_fine

            if args.rdrop_alpha > 0.0:
                batch_kl_loss_fine = float(ori_loss['kl_loss'].item())
                batch_kl_loss += batch_kl_loss_fine
                epoch_kl_loss += batch_kl_loss_fine

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

        last_c_loss.append(batch_c_loss)
        last_loss.append(batch_loss)
        if args.rdrop_alpha > 0.0:
            last_kl_loss.append(batch_kl_loss)
            if len(last_kl_loss) > 50:
                last_kl_loss.pop(0)
        if len(last_c_loss) > 50:
            last_c_loss.pop(0)
        if len(last_loss) > 50:
            last_loss.pop(0)

        if accelerator.is_local_main_process:
            if args.rdrop_alpha > 0.0:
                epoch_iterator.set_description("Epoch: %0.4f/%0.4f/%0.4f/%0.4f, Batch: %0.4f/%0.4f/%0.4f/%0.4f" % \
                                               (epoch_loss / (batch_idx + 1), epoch_kl_loss / (batch_idx + 1),
                                                epoch_c_loss / (batch_idx + 1), epoch_alpha_loss / (batch_idx + 1),\
                                                batch_loss, batch_kl_loss, batch_c_loss, batch_alpha_loss)
                                               )
                if steps %50 == 0:
                    if scheduler is not None:
                        lrs = scheduler.get_lr()
                    else:
                        lrs = [args.learning_rate]
                    #print('W:', model.decoder.W.weight.detach()[0].cpu())
                    if not isinstance(model, DistributedDataParallel):
                        if hasattr(model.decoder, 'code_alphas'):
                            wandb.log({'train/train_loss': np.mean(last_loss[-50:]), 'train/train_kl_loss': np.mean(last_kl_loss[-50:]),
                                   'train/train_c_loss': np.mean(last_c_loss[-50:]), 'train/train_alpha_loss': epoch_alpha_loss / (batch_idx + 1),
                                       'alphas/weight': model.decoder.code_alphas.detach().cpu().mean(dim=0), 'train/lr': lrs[0]})
                        else:
                            wandb.log({'train/train_loss': np.mean(last_loss[-50:]),
                                       'train/train_kl_loss': np.mean(last_kl_loss[-50:]),
                                       'train/train_c_loss': np.mean(last_c_loss[-50:]),
                                       'train/train_alpha_loss': epoch_alpha_loss / (batch_idx + 1), 'train/lr': lrs[0]})
                    else:
                        if hasattr(model.module.decoder, 'code_alphas'):
                            wandb.log({'train/train_loss': np.mean(last_loss[-50:]), 'train/train_kl_loss': np.mean(last_kl_loss[-50:]),
                                   'train/train_c_loss': np.mean(last_c_loss[-50:]), 'train/train_alpha_loss': epoch_alpha_loss / (batch_idx + 1),
                                       'alphas/weight': model.module.decoder.code_alphas.detach().cpu().mean(dim=0), 'train/lr': lrs[0]})
                        else:
                            wandb.log({'train/train_loss': np.mean(last_loss[-50:]),
                                       'train/train_kl_loss': np.mean(last_kl_loss[-50:]),
                                       'train/train_c_loss': np.mean(last_c_loss[-50:]),
                                       'train/train_alpha_loss': epoch_alpha_loss / (batch_idx + 1), 'train/lr': lrs[0]})
            else:
                epoch_iterator.set_description("Epoch: %0.4f, Batch: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss))
                if steps % 50 == 0:
                    wandb.log({'train/train_loss': epoch_loss / (batch_idx + 1), 'train/train_alpha_loss': epoch_alpha_loss / (batch_idx + 1)})

        if (steps + 1) % args.gradient_accumulation_steps == 0:
            try:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            except Exception as e:
                print(e)
                print("find nan or inf in gradients")
                print(loss)
                print(ori_loss['kl_loss'])
                print(ori_loss['c_loss'])
                print(model.decoder.code_alphas.detach().cpu().mean(dim=0))
                with torch.autograd.detect_anomaly():
                    ori_loss = model(batch_gpu, rdrop=args.rdrop_alpha > 0.0)
                    loss = ori_loss['loss']
                    loss.backward()
                exit(-1)
            optimizer.step()
            model.zero_grad()
        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule
        steps += 1

    tqdm_bar = True
    if accelerator.is_local_main_process:
        dev_metric, _, threshold = eval_func(model, dev_dataloader, accelerator.device, None, tqdm_bar, args)
        print('Threshold find on dev:', np.mean(threshold))
        test_metric, _, _ = eval_func(model, test_dataloader, accelerator.device, threshold, tqdm_bar, args)
        keys = dev_metric.keys()
        dev_metric_new = {}
        for key in keys:
            dev_metric_new['dev/' + key] = dev_metric[key]
        wandb.log(dev_metric_new)
        keys = test_metric.keys()
        test_metric_new = {}
        for key in keys:
            test_metric_new['test/' + key] = test_metric[key]
        wandb.log(test_metric)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        dev_metric = None
        test_metric = None
    return dev_metric, test_metric, steps

def predict(model, dataloader, device, threshold=None, tqdm_bar=None, args=None):
    model.eval()
    outputs = []
    device = args.device if args is not None else device
    it = tqdm(dataloader) if tqdm_bar else dataloader
    with torch.no_grad():
        if isinstance(model, DistributedDataParallel):
            model.module.calculate_label_hidden()
            if args.use_graph:
                model.module.calculate_label_hidden_m()
            else:
                model.module.mlabel_feats = None
            if args.predict_with != 'default':
                model.module.decoder.text_pooling = args.predict_with
                model.module.decoder.head_pooling = args.predict_with
        else:
            model.calculate_label_hidden()
            if args.use_graph:
                model.calculate_label_hidden_m()
            else:
                model.mlabel_feats = None
            if args.predict_with != 'default':
                model.decoder.text_pooling = args.predict_with
                model.decoder.head_pooling = args.predict_with
        for batch in it:
            batch_gpu = tuple([x.to(device) for x in batch])
            if isinstance(model, DistributedDataParallel):
                now_res = model.module.predict(batch_gpu, threshold)
            else:
                now_res = model.predict(batch_gpu, threshold)
            outputs.append({key:value for key, value in now_res.items()})
        if isinstance(model, DistributedDataParallel):
            if args.predict_with != 'default':
                model.module.decoder.text_pooling = args.text_pooling
                model.module.decoder.head_pooling = args.head_pooling
        else:
            if args.predict_with != 'default':
                model.decoder.text_pooling = args.text_pooling
                model.decoder.head_pooling = args.head_pooling

            
    yhat = np.concatenate([output['yhat'] for output in outputs])
    yhat_raw = np.concatenate([output['yhat_raw'] for output in outputs])
    y = np.concatenate([output['y'] for output in outputs])
    return yhat, y, yhat_raw

def eval_func(model, dataloader, device, threshold=None, tqdm_bar=False, args=None):
    yhat, y, yhat_raw = predict(model, dataloader, device, threshold, tqdm_bar, args)
    if threshold is None:
        if args.threshold_find == 'v1':
            threshold = find_threshold_micro(yhat_raw, y)
        elif args.threshold_find == 'v2':
            threshold = find_threshold_micro_v2(yhat_raw, y)
    yhat = np.where(yhat_raw > threshold, 1, 0)
    metric = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
    return metric, (yhat, y, yhat_raw), threshold

def main():
    torch.manual_seed(1234)
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
