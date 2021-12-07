import argparse
import math
import os
import yaml
import warnings

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader, get_dataset
from models.MBNet import MBNet
from models.LDNet import LDNet
from models.loss import Loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from inference import save_results

writer = SummaryWriter()
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid(mode, model, dataloader, systems, save_dir, steps, prefix):
    model.eval()

    predict_mean_scores = []
    true_mean_scores = []
    predict_sys_mean_scores = {system:[] for system in systems}
    true_sys_mean_scores = {system:[] for system in systems}
    
    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        mag_sgrams_padded, avg_scores, sys_names, wav_names = batch
        mag_sgrams_padded = mag_sgrams_padded.to(device)

        # forward
        with torch.no_grad():
            try:
                # actual inference
                if mode == "mean_net":
                    pred_mean_scores = model.only_mean_inference(spectrum = mag_sgrams_padded)
                elif mode == "all_listeners":
                    pred_mean_scores, _ = model.average_inference(spectrum = mag_sgrams_padded)
                elif mode == "mean_listener":
                    pred_mean_scores = model.mean_listener_inference(spectrum = mag_sgrams_padded)
                else:
                    raise NotImplementedError

                pred_mean_scores = pred_mean_scores.cpu().detach().numpy()
                predict_mean_scores.extend(pred_mean_scores.tolist())
                true_mean_scores.extend(avg_scores.tolist())
                for j, sys_name in enumerate(sys_names):
                    predict_sys_mean_scores[sys_name].append(pred_mean_scores[j])
                    true_sys_mean_scores[sys_name].append(avg_scores[j])
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # print(f"[Runner] - CUDA out of memory at step {global_step}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    predict_mean_scores = np.array(predict_mean_scores)
    true_mean_scores = np.array(true_mean_scores)
    predict_sys_mean_scores = np.array([np.mean(scores) for scores in predict_sys_mean_scores.values()])
    true_sys_mean_scores = np.array([np.mean(scores) for scores in true_sys_mean_scores.values()])
    
    utt_MSE=np.mean((true_mean_scores-predict_mean_scores)**2)
    utt_LCC=np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    utt_SRCC=scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    utt_KTAU=scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    sys_MSE=np.mean((true_sys_mean_scores-predict_sys_mean_scores)**2)
    sys_LCC=np.corrcoef(true_sys_mean_scores, predict_sys_mean_scores)[0][1]
    sys_SRCC=scipy.stats.spearmanr(true_sys_mean_scores, predict_sys_mean_scores)[0]
    sys_KTAU=scipy.stats.kendalltau(true_sys_mean_scores, predict_sys_mean_scores)[0]
    
    print(
        f"\n[{prefix}][{steps}][UTT][ MSE = {utt_MSE:.4f} | LCC = {utt_LCC:.4f} | SRCC = {utt_SRCC:.4f} ] [SYS][ MSE = {sys_MSE:.4f} | LCC = {sys_LCC:.4f} | SRCC = {sys_SRCC:.4f} ]\n"
    )
        
    save_results(steps, [utt_MSE, utt_LCC, utt_SRCC, utt_KTAU], [sys_MSE, sys_LCC, sys_SRCC, sys_KTAU], os.path.join(save_dir, "training_" + mode + ".csv"))

    torch.save(model.state_dict(), os.path.join(save_dir, f"model-{steps}.pt"))
    model.train()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default = "vcc2018")
    parser.add_argument("--data_dir", type=str, default = "data/vcc2018")
    parser.add_argument("--exp_dir", type=str, default = "exp")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--update_freq", type=int, default=1,
                        help="If GPU OOM, decrease the batch size and increase this.")
    parser.add_argument('--seed', default=1337, type=int)
    args = parser.parse_args()
    
    # Fix seed and make backends deterministic
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # because we have dynamic input size

    # fix issue of too many opened files
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
        
    # read config
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print("[Info] LR: {}".format(config["optimizer"]["lr"]))
    print("[Info] alpha: {}".format(config["alpha"]))
    print("[Info] lambda: {}".format(config["lambda"]))

    # define and make dirs
    save_dir = os.path.join(args.exp_dir, args.tag)
    os.makedirs(save_dir, exist_ok=True)
    idtable_path = os.path.join(save_dir, "idtable.pkl")

    # define dataloders
    train_set = get_dataset(args.dataset_name, args.data_dir, "train", idtable_path, config["padding_mode"], config["use_mean_listener"])
    valid_set = get_dataset(args.dataset_name, args.data_dir, "valid", idtable_path)
    train_loader = get_dataloader(train_set, batch_size=config["train_batch_size"], num_workers=6)
    valid_loader = get_dataloader(valid_set, batch_size=config["test_batch_size"], num_workers=1, shuffle=False)
    print("[Info] Number of training samples: {}".format(len(train_set)))
    print("[Info] Number of validation samples: {}".format(len(valid_set)))
    
    # get number of judges
    num_judges = train_set.num_judges
    config["num_judges"] = num_judges
    print("[Info] Number of judges: {}".format(num_judges))
    print("[Info] Use mean listener: {}".format("True" if config["use_mean_listener"] else "False"))

    # define model
    if config["model"] == "MBNet":
        model = MBNet(config).to(device)
    elif config["model"] == "LDNet":
        model = LDNet(config).to(device)
    else:
        raise NotImplementedError
    print("[Info] Model parameters: {}".format(model.get_num_params()))
    criterion = Loss(config["output_type"], config["alpha"], config["lambda"], config["tau"], config["mask_loss"])

    # optimizer
    optimizer = get_optimizer(model, config["total_steps"], config["optimizer"])
    optimizer.zero_grad()

    # scheduler
    scheduler = None
    if config.get('scheduler'):
        scheduler = get_scheduler(optimizer, config["total_steps"], config["scheduler"])

    # set pbar
    pbar = tqdm(total=config["total_steps"], ncols=0, desc="Overall", unit=" step")

    # count accumulated gradients
    backward_steps = 0

    # write config
    with open(os.path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # actual training loop
    model.train()
    while pbar.n < pbar.total:
        for i, batch in enumerate(
            tqdm(train_loader, ncols=0, desc="Train", unit=" step")
        ):
            try:
                if pbar.n >= pbar.total:
                    break
                global_step = pbar.n + 1

                # fetch batch and put on device
                mag_sgrams_padded, mag_sgrams_lengths, avg_scores, scores, judge_ids = batch
                mag_sgrams_padded = mag_sgrams_padded.to(device)
                judge_ids = judge_ids.to(device)
                avg_scores = avg_scores.to(device)
                scores = scores.to(device)

                # forward
                # each has shape [batch, time, 1 (scalar) / 5 (categorical)]
                pred_mean_scores, pred_ld_scores = model(spectrum = mag_sgrams_padded, 
                                                         judge_id = judge_ids,
                                                         )
                
                # loss calculation
                loss, mean_loss, ld_loss = criterion(pred_mean_scores, avg_scores, pred_ld_scores, scores, mag_sgrams_lengths, device)

                (loss / args.update_freq).backward()

                if config["alpha"] > 0:
                    pbar.set_postfix(
                        {
                            "loss": loss.item(),
                            "mean_loss": mean_loss.item(),
                            "LD_loss": ld_loss.item(),
                        }
                    )
                else:
                    pbar.set_postfix(
                        {
                            "loss": loss.item(),
                            "LD_loss": ld_loss.item(),
                        }
                    )

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[Runner] - CUDA out of memory at step {global_step}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise
            
            # release GPU memory
            del loss

            # whether to accumulate gradient
            backward_steps += 1
            if backward_steps % args.update_freq > 0:
                continue

            # gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])

            # optimize
            if math.isnan(grad_norm):
                print(f"[Runner] - grad norm is NaN at step {global_step}")
            else:
                optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if scheduler:
                scheduler.step()

            # evaluate
            if global_step % config["valid_steps"] == 0:
                valid(config["inference_mode"], model, valid_loader, valid_set.systems, save_dir, global_step, "Valid")
            pbar.update(1)
    
if __name__ == "__main__":
    main()
