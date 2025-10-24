import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch import nn
from torch.nn import functional as F
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--loss_method', type=str, default='mse')
parser.add_argument('--run_name', type=str, default='mmd_ddpm')
parser.add_argument('--sampling', action='store_true')
parser.add_argument('--sampling_ckpt', type=int, default=None)
args = parser.parse_args()

if args.loss_method == 'mse':
    loss_method = 'mse'
elif args.loss_method == 'mmd':
    loss_method = 'mmd'
elif args.loss_method == 'mmd-hessian':
    loss_method = 'mmd-hessian'
elif args.loss_method == 'mmd-hessian-hvp':
    loss_method = 'mmd-hessian-hvp'
else:
    raise ValueError(f'unknown loss method {args.loss_method}')

class Cross_DiT(nn.Module):
    """
    Minimal Cross_DiT-style module with fixed configuration:
    - hidden size: 128
    - transformer depth: 2
    - attention heads: 2

    Expected inputs:
        x_input: (batch, S, L)
        t:       (batch, 1)
    """
    def __init__(self, length: int):
        super().__init__()
        self.channels = 1
        self.self_condition = False
        hidden_size = 128

        self.input_proj = nn.Linear(length, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=2,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_size, length)

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_input.dim() != 3:
            raise ValueError("x_input must have shape (batch, S, L)")
        if t.dim() != 2 or t.size(1) != 1:
            raise ValueError("t must have shape (batch, 1)")
        if x_input.size(0) != t.size(0):
            raise ValueError("batch dimension of x_input and t must match")

        # Project inputs and add broadcasted time embedding
        x = self.input_proj(x_input)
        time_embed = self.time_mlp(t).unsqueeze(1)
        x = x + time_embed

        x = self.blocks(x)
        return self.output_proj(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.channels = 1
        self.self_condition = False
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dim+1, hidden_dim))
            input_dim = hidden_dim - 1
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        x= x.squeeze(1) # 256,1,2000 -> 256,2000
        #x = x.view(1, -1) # 1, batch * 2000, # [batch] 1, batch
        xt = torch.cat([x, t.unsqueeze(1)], dim = 1)
        for layer in self.layers:
            xt = layer(xt)
            xt = F.relu(xt)
        x = self.output_layer(xt)
        return x.reshape(-1, 2000).unsqueeze(1)#.squeeze(0).unsqueeze(1)

model = Cross_DiT(length = 2000)
#print num model parameter 
print(sum(p.numel() for p in model.parameters()))

'''
model = MLP(
    input_dim = 2000,
    hidden_dim = 128,
    output_dim = 2000,
    num_layers = 3,
)
'''

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 2000,
    timesteps = 1000,
    objective = 'pred_x0'
)
diffusion.loss_method = loss_method
#diffusion.is_ddim_sampling = False
#diffusion.sampling_timesteps = 1000

training_seq = torch.zeros(1901, 1, 2000)
for i in range(1901):
    training_seq[i, :, i:i+100] = 1

for x in training_seq.squeeze(1):
    assert (x!=0).sum() == 100

if __name__ == "__main__":

    if not args.sampling:
        wandb_run = wandb.init(
            project="mmd_score_matching",
            name=args.run_name,
            config={
                "loss_method": loss_method,
                "train_batch_size": 256,
                "train_lr": 8e-5,
                "train_num_steps": 50000,
                "ema_decay": 0.995,
                "sequence_length": 2000,
                "timesteps": 1000,
            },
        )
    else:
        wandb_run = None

    trainer = Trainer1D(
        diffusion,
        dataset = training_seq,
        train_batch_size = 256,
        train_lr = 8e-5,
        train_num_steps = 50000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                       # turn on mixed precision
        save_and_sample_every = 50000,
        wandb_run = wandb_run,
    )
    if args.sampling:
        assert args.sampling_ckpt is not None 
        trainer.load(args.sampling_ckpt)
        sampled_seq = diffusion.sample(batch_size = 256)
        torch.save(sampled_seq, 'sampled_seq.pt')
    else:
        trainer.train()
        sampled_seq = diffusion.sample(batch_size = 256)
        torch.save(sampled_seq, 'sampled_seq.pt')
    if wandb_run is not None:
        wandb_run.finish()
    exit()
    # 10 <- batch size mmd 
    # 99 <- batch size mse 
    # 128 <- gene size mmd 
    # 256 <- gene size mse
    # 9999 <- MSE 50k 
    # 8888 <- MMD 50k 
    # 7777 <- MMD-HESSIAN 50k