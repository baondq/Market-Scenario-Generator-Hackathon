import yaml
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import convert_dict_to_tuple, EarlyStopper
from data_utils.custom_dataloader import get_dataloaders
from diffusion.trainer import DiffusionTrainer


def train(config_path:str):

    with open(config_path, "r")  as f:
        configs = convert_dict_to_tuple(yaml.safe_load(f))
    
    train_loader, eval_loader = get_dataloaders(configs)

    trainer = DiffusionTrainer(configs)

    stop_checker = EarlyStopper(patience=configs.train_configs.patience)

    step = 0
    best_eval_loss = float("inf")
    best_eval_rmspe = float("inf")
    progress_bar = tqdm(range(configs.train_configs.num_training_steps))
    train_loss_log = []
    eval_loss_log = []
    eval_rmspe_log = []

    while step < configs.train_configs.num_training_steps:

        for data in train_loader:
            x_t, y, t, noise = data
            train_loss_step = trainer.train_one_step(x_t, y, t, noise)
            train_loss_log.append(train_loss_step)
            step += 1

            if step > len(train_loader):
                progress_bar.set_postfix({'lr': trainer.scheduler.get_last_lr()[0], 'train_step_loss': train_loss_step, 'val_loss': eval_loss, 'val_rmspe': eval_rmspe})
            else:
                progress_bar.set_postfix({'lr': trainer.scheduler.get_last_lr()[0], 'train_step_loss': train_loss_step})
            progress_bar.update(1)
        
        eval_loss, eval_rmspe = trainer.evaluate(eval_loader)
        eval_loss_log.append(eval_loss)
        eval_rmspe_log.append(eval_rmspe)

        stop_check = stop_checker.early_stop(eval_loss)

        if eval_loss < best_eval_loss:
            best_state_dict = trainer.denoiser.state_dict()
            best_eval_loss = eval_loss
            best_eval_rmspe = eval_rmspe
        
        if stop_check:
            break
    
    torch.save(best_state_dict, configs.train_configs.state_dict_out_path)
    print(f"best evaluation mse loss = {best_eval_loss} & corresponding evaluation rmspe = {best_eval_rmspe}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(train_loss_log)), train_loss_log,label="train loss step")
    ax.plot(len(train_loader)*np.arange(len(eval_loss_log)), eval_loss_log,label="evaluation loss")
    ax.plot(len(train_loader)*np.arange(len(eval_rmspe_log)), eval_rmspe_log,label="evaluation RMSPE")
    ax.legend()
    plt.show()


if __name__ == "__main__":

    train("configs/exp4.yml")
