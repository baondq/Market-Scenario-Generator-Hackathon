import torch
from torch.utils.data import DataLoader, random_split
import pickle

from data_utils.custom_dataset import SampleDataset


def get_dataloaders(configs):
    with open(configs.data_configs.data_path, "rb") as f:
        sample_data = pickle.load(f)
    with open(configs.data_configs.label_path, "rb") as f:
        sample_label = pickle.load(f)
    sample_dataset = SampleDataset(data=sample_data, label=sample_label, configs=configs)
    sample_train, sample_eval = random_split(sample_dataset, [0.8,0.2], generator = torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        sample_train,
        batch_size = configs.data_configs.batch_size,
        shuffle = True,
        drop_last = False
    )
    eval_loader = DataLoader(
        sample_eval,
        batch_size = configs.data_configs.batch_size,
        shuffle = False,
        drop_last = False
    )
    return train_loader, eval_loader


# unit test
if __name__ == "__main__":

    import yaml
    from utils.utils import convert_dict_to_tuple

    with open("configs/exp1.yml", "r")  as f:
        configs = convert_dict_to_tuple(yaml.safe_load(f))
    
    train_loader, eval_loader = get_dataloaders(configs)
    for d in train_loader:
        x_t, y, t, noise = d
        print(x_t.size(), y.size(), t.size(), noise.size())
        break
