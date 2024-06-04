import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import numpy as np

from base_models.modified_lstm import ModifiedLSTM


class DiffusionTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.denoiser = ModifiedLSTM(configs)
        self.optimizer = AdamW(
            params = self.denoiser.parameters(),
            lr = configs.train_configs.lr
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer = self.optimizer,
            num_warmup_steps = configs.train_configs.num_warmup_steps,
            num_training_steps = configs.train_configs.num_training_steps
        )
    
    def train_one_step(self, x_t, y, t, label):
        self.denoiser.train()
        self.optimizer.zero_grad()

        if self.configs.train_configs.train_unconditional:
            if torch.rand(1)[0] < 1/10:
                y = None
        
        x_t = x_t.to(self.configs.model_configs.device)
        if y is not None:
            y = y.to(self.configs.model_configs.device)
        t = t.to(self.configs.model_configs.device)
        label = label.to(self.configs.model_configs.device)
        
        et = self.denoiser(x_t, y, t)
        loss_dm = F.mse_loss(et, label, reduction='none')

        if y is not None:
            y_bal = torch.from_numpy(np.random.choice(self.configs.model_configs.num_cls, size=(x_t.size(0),), p=self.configs.train_configs.cls_weights)).to(x_t.device).type(torch.int32)
            et_bal = self.denoiser(x_t, y_bal, t)
            weight = t[:, None, None, None] / self.configs.model_configs.max_steps * self.configs.train_configs.tau
            loss_reg = weight * F.mse_loss(et, et_bal.detach(), reduction='none')
            loss_com = weight * F.mse_loss(et.detach(), et_bal, reduction='none')
            loss = loss_dm + loss_reg + self.configs.train_configs.gamma * loss_com
        else:
            loss = loss_dm
        
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.configs.model_configs.clip_value)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
    
    def evaluate(self, eval_loader):
        eval_loss = 0
        eval_rmspe = 0
        self.denoiser.eval()
        for data in eval_loader:
            x_t, y, t, label = data
            x_t = x_t.to(self.configs.model_configs.device)
            y = y.to(self.configs.model_configs.device)
            t = t.to(self.configs.model_configs.device)
            label = label.to(self.configs.model_configs.device)
            with torch.no_grad():
                et = self.denoiser(x_t, y, t)
                loss = F.mse_loss(et, label, reduction="none")
                eval_loss += loss.mean().item()
                eval_rmspe += torch.sqrt(torch.mean(torch.clip(loss / label**2, max = 5))).item()
        return eval_loss / len(eval_loader), eval_rmspe / len(eval_loader)


# unit test
if __name__ == "__main__":

    import yaml
    from utils.utils import convert_dict_to_tuple

    with open("configs/exp1.yml", "r")  as f:
        configs = convert_dict_to_tuple(yaml.safe_load(f))
    
    y = torch.from_numpy(np.random.choice(configs.model_configs.num_cls, size=(64,), p=configs.model_configs.cls_weights))
    print(y)
