import torch
import torch.nn as nn

from utils.utils import get_timestep_embedding


class ModifiedLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.label_emb = nn.Embedding(
            num_embeddings = configs.model_configs.num_cls,
            embedding_dim = configs.model_configs.emb_dim
        )
        self.feature_fc = nn.Linear(
            in_features = configs.model_configs.feature_dim,
            out_features = configs.model_configs.emb_dim
        )
        self.label_fc = nn.Linear(
            in_features = configs.model_configs.emb_dim,
            out_features = configs.model_configs.emb_dim
        )
        self.step_fc = nn.Linear(
            in_features = configs.model_configs.emb_dim,
            out_features = configs.model_configs.emb_dim
        )
        num_layers = configs.model_configs.num_layers or 1
        self.lstm = nn.LSTM(
            input_size = configs.model_configs.emb_dim,
            hidden_size = configs.model_configs.hid_dim,
            batch_first = True,
            bidirectional = configs.model_configs.bidirectional,
            num_layers = num_layers
        )
        D = 2 if configs.model_configs.bidirectional else 1
        self.out_fc = nn.Linear(
            in_features = D * configs.model_configs.hid_dim,
            out_features = configs.model_configs.feature_dim
        )
    
    def forward(self, x_t, y, t):
        t_emb = get_timestep_embedding(timesteps=t, embedding_dim=self.configs.model_configs.emb_dim, max_period=self.configs.model_configs.max_steps)
        t_emb = torch.unsqueeze(t_emb, dim=1)
        t_emb = t_emb.expand((t_emb.size(0), x_t.size(1), t_emb.size(2)))
        if y is not None:
            y_emb = self.label_emb(y)
            y_emb = torch.unsqueeze(y_emb, dim=1)
            y_emb = y_emb.expand((y_emb.size(0), x_t.size(1), y_emb.size(2)))
            out = self.feature_fc(x_t) + self.label_fc(y_emb) + self.step_fc(t_emb)
        else:
            out = self.feature_fc(x_t) + self.step_fc(t_emb)
        out, _ = self.lstm(out)
        out = self.out_fc(out)
        if self.configs.model_configs.predict_noise or self.configs.data_configs.preprocess:
            return out
        else:
            out[:,:,::2] = torch.tanh(out[:,:,::2])
            out[:,:,1::2] = torch.sigmoid(out[:,:,1::2])
            return out


# unit test
if __name__ == "__main__":

    import yaml
    from utils.utils import convert_dict_to_tuple

    with open("configs/exp3.yml", "r")  as f:
        configs = convert_dict_to_tuple(yaml.safe_load(f))

    model = ModifiedLSTM(configs)
    x_t = torch.rand((64,5,10))
    # y = torch.randint(low=0, high=2, size=(64,))
    y = None
    t = torch.randint(low=0, high=1001, size=(64,))
    out = model(x_t, y, t)
    print(out.size())

