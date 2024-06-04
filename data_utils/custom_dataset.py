import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import math

from utils.utils import sigmoid


class SampleDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray, configs):
        super().__init__()
        self.data = data
        self.label = label
        self.configs = configs
        self.T = configs.model_configs.max_steps
        with open(configs.data_configs.stats_path, "rb") as f:
            self.stats = pickle.load(f)
        if configs.data_configs.preprocess:
            self.preprocess_data()          # standardize data
        

    def __len__(self):
        return self.data.shape[0]
    
    def preprocess_data(self):
        self.data[self.label[:,0]==0,:,::2] = (self.data[self.label[:,0]==0,:,::2] - self.stats["log_reg_mean"]) / self.stats["log_reg_std"]
        self.data[self.label[:,0]==0,:,1::2] = (self.data[self.label[:,0]==0,:,1::2] - self.stats["vol_reg_mean"]) / self.stats["vol_reg_std"]
        self.data[self.label[:,0]==1,:,::2] = (self.data[self.label[:,0]==1,:,::2] - self.stats["log_cri_mean"]) / self.stats["log_cri_std"]
        self.data[self.label[:,0]==1,:,1::2] = (self.data[self.label[:,0]==1,:,1::2] - self.stats["vol_cri_mean"]) / self.stats["vol_cri_std"]
    
    @staticmethod
    def get_linear_schedule(t:float, clip_min=1e-16):
        return np.clip(1 - t, clip_min, 1.)
    
    @staticmethod
    def get_sigmoid_schedule(t:float, start=-3, end=3, tau=1.0, clip_min=1e-16):
        v_start = sigmoid(start / tau)
        v_end = sigmoid(end / tau)
        output = sigmoid((t * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start)
        return np.clip(output, clip_min, 1.)
    
    @staticmethod
    def get_cosine_schedule(t:float, start=0, end=1, tau=1, clip_min=1e-16):
        v_start = math.cos(start * math.pi / 2) ** (2 * tau)
        v_end = math.cos(end * math.pi / 2) ** (2 * tau)
        output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
        output = (v_end - output) / (v_end - v_start)
        return np.clip(output, clip_min, 1.)

    def get_noise_prob(self, t:int):
        method = self.configs.noise_configs.method
        if method == "cosine":
            alpha = self.get_cosine_schedule(t/self.T, tau=self.configs.noise_configs.cosine.tau)
        if method == "linear":
            alpha = self.get_linear_schedule(t/self.T)
        if method == "sigmoid":
            alpha = self.get_sigmoid_schedule(t/self.T, tau=self.configs.noise_configs.sigmoid.tau)
        return alpha

    def __getitem__(self, idx):
        t = np.random.randint(low=1, high=self.T+1)
        alpha = self.get_noise_prob(t)
        x_0 = torch.from_numpy(self.data[idx,:,:])
        y = torch.tensor(self.label[idx,0], dtype=torch.int32)
        t = torch.tensor(t, dtype=torch.int32)
        epsilon_list = []
        for _ in range(x_0.shape[0]):
            if self.label[idx,0] == 0:
                cov = self.stats["reg_corr"] if self.configs.data_configs.preprocess else self.stats["reg_cov"]
            elif self.label[idx,0] == 1:
                cov = self.stats["cri_corr"] if self.configs.data_configs.preprocess else self.stats["cri_cov"]
            epsilon = np.random.multivariate_normal(mean=np.zeros((x_0.shape[1],)), cov=cov)
            epsilon_list.append(epsilon)
        noise = torch.from_numpy(np.stack(epsilon_list))
        x_t = math.sqrt(alpha) * x_0 + math.sqrt(1-alpha) * noise
        if self.configs.model_configs.predict_noise:
            return x_t.type(torch.float32), y, t, noise.type(torch.float32)
        else:
            return x_t.type(torch.float32), y, t, x_0.type(torch.float32)


# unit test
if __name__ == "__main__":

    from utils.utils import convert_dict_to_tuple
    import yaml

    with open("data/ref_data.pkl", "rb") as f:
        ref_data = pickle.load(f)

    with open("data/ref_label.pkl", "rb") as f:
        ref_label = pickle.load(f)
    
    with open("configs/exp1.yml", "r") as f:
        configs = convert_dict_to_tuple(yaml.safe_load(f))
    
    sample_dataset = SampleDataset(data=ref_data, label=ref_label, configs=configs)
    x_t, y, t, noise = sample_dataset.__getitem__(0)
    print(x_t)
    print(y)
    print(t)
    print(noise)

    with open("data/stats.pkl", "rb") as f:
        stats = pickle.load(f)
    print(stats)
