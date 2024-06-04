import torch
import numpy as np
import math
import pickle
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal

from base_models.modified_lstm import ModifiedLSTM


class DiffusionSampler:
    def __init__(self, configs):
        self.configs = configs
        self.T = configs.model_configs.max_steps
        self.denoiser = ModifiedLSTM(configs)
        self.denoiser.load_state_dict(torch.load(configs.model_configs.state_dict_path))
        with open(configs.data_configs.stats_path, "rb") as f:
            self.stats = pickle.load(f)
    
    @staticmethod
    def get_cosine_schedule(t:float, start=0, end=1, tau=1, clip_min=1e-9):
        v_start = math.cos(start * math.pi / 2) ** (2 * tau)
        v_end = math.cos(end * math.pi / 2) ** (2 * tau)
        output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
        output = (v_end - output) / (v_end - v_start)
        return output

    def get_noise_prob(self, t:int):
        alpha = self.get_cosine_schedule(t/self.T, tau=self.configs.noise_configs.cosine.tau)
        return alpha
    
    def compute_q_posterior_mean(self, t:int, x_0: torch.Tensor, x_t: torch.Tensor):
        assert t > 0
        alpha = self.get_noise_prob(t)
        alpha_prev = self.get_noise_prob(t-1)
        beta = 1 - alpha / alpha_prev
        coeff_0 = math.sqrt(alpha_prev) * beta / (1 - alpha)
        coeff_t = math.sqrt(alpha) * (1 - alpha_prev) / (1 - alpha)
        return coeff_0 * x_0 + coeff_t * x_t
    
    def compute_q_posterior_cov_factor(self, t:int):
        assert t > 0
        alpha = self.get_noise_prob(t)
        alpha_prev = self.get_noise_prob(t-1)
        beta = 1 - alpha / alpha_prev
        return beta * (1 - alpha_prev) / (1 - alpha)
    
    def sample_one_step(self, t:int, x_t:torch.Tensor, y:torch.Tensor):
        self.denoiser.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t).expand(y.size(0)).type(torch.int32)
            if self.configs.model_configs.predict_noise:
                eps = self.denoiser(x_t, y, t_tensor)
                eps_unc = self.denoiser(x_t, None, t_tensor)
                eps = eps + self.configs.sampling_configs.guidance_strength * (eps - eps_unc)
                alpha = self.get_noise_prob(t)
                x_0 = (x_t - math.sqrt(1-alpha) * eps) / math.sqrt(np.clip(alpha, 1e-9, 1.0))
            else:
                x_0 = self.denoiser(x_t, y, t_tensor)
                x0_unc = self.denoiser(x_t, None, t_tensor)
                x_0 = x_0 + self.configs.sampling_configs.guidance_strength * (x_0 - x0_unc)
            if t == 1:
                return x_0
            else:
                x_new_list = []
                q_posterior_cov_factor = self.compute_q_posterior_cov_factor(t)
                for i in range(y.size(0)):
                    q_posterior_mean = self.compute_q_posterior_mean(t, x_0[i,:,:], x_t[i,:,:])
                    if self.configs.data_configs.preprocess:
                        cov = self.stats["reg_corr"] if y[i] == 0 else self.stats["cri_corr"]
                    else:
                        cov = self.stats["reg_cov"] if y[i] == 0 else self.stats["cri_cov"]
                    cov = torch.from_numpy(cov).type(torch.float32)
                    gaussian_sampler = MultivariateNormal(q_posterior_mean, q_posterior_cov_factor*cov)
                    x_new_list.append(gaussian_sampler.sample())
                x_new = torch.stack(x_new_list, dim=0)
                return x_new
    
    def sample(self, condition: torch.Tensor):
        ts_length = self.configs.sampling_configs.ts_length
        feature_dim = self.configs.sampling_configs.feature_dim
        y = torch.squeeze(condition, dim=1)
        x_t_list = []
        for i in range(y.size(0)):
            if self.configs.data_configs.preprocess:
                cov = self.stats["reg_corr"] if y[i] == 0 else self.stats["cri_corr"]
            else:
                cov = self.stats["reg_cov"] if y[i] == 0 else self.stats["cri_cov"]
            cov = torch.from_numpy(cov).type(torch.float32)
            gaussian_sampler = MultivariateNormal(torch.zeros(ts_length, feature_dim), cov)
            x_t_list.append(gaussian_sampler.sample())
        x_t = torch.stack(x_t_list, dim=0)
        for t in tqdm(reversed(range(1, self.T+1)), total=self.T):
            x_t = self.sample_one_step(t, x_t, y)
        if self.configs.data_configs.preprocess:
            for i in range(y.size(0)):
                if y[i] == 0:
                    x_t[i,:,::2] = x_t[i,:,::2] * torch.from_numpy(self.stats["log_reg_std"]).type(torch.float32) + torch.from_numpy(self.stats["log_reg_mean"]).type(torch.float32)
                    x_t[i,:,1::2] = x_t[i,:,1::2] * torch.from_numpy(self.stats["vol_reg_std"]).type(torch.float32) + torch.from_numpy(self.stats["vol_reg_mean"]).type(torch.float32)
                elif y[i] == 1:
                    x_t[i,:,::2] = x_t[i,:,::2] * torch.from_numpy(self.stats["log_cri_std"]).type(torch.float32) + torch.from_numpy(self.stats["log_cri_mean"]).type(torch.float32)
                    x_t[i,:,1::2] = x_t[i,:,1::2] * torch.from_numpy(self.stats["vol_cri_std"]).type(torch.float32) + torch.from_numpy(self.stats["vol_cri_mean"]).type(torch.float32)
        x_t[:,:,1::2] = torch.clip(x_t[:,:,1::2], min=0)
        return x_t


if __name__ == "__main__":

    import yaml
    import matplotlib.pyplot as plt
    from utils.utils import convert_dict_to_tuple

    with open("configs\sampler_exp4.yml", "r")  as f:
        configs = convert_dict_to_tuple(yaml.safe_load(f))
    
    sampler = DiffusionSampler(configs)
    regular_condition = torch.zeros(configs.sampling_configs.batch_size, 1).type(torch.int32)
    crisis_condition = torch.ones(configs.sampling_configs.batch_size, 1).type(torch.int32)
    regular_ts = sampler.sample(regular_condition).cpu().numpy()
    crisis_ts = sampler.sample(crisis_condition).cpu().numpy()

    with open("data/fake_regular_exp4.pkl", "wb") as f:
        pickle.dump(regular_ts, f)
    
    with open("data/fake_crisis_exp4.pkl", "wb") as f:
        pickle.dump(crisis_ts, f)

