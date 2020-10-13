#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import logging
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import slayerSNN as snn
from dataset import ViTacDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()

parser = argparse.ArgumentParser("Train AE models.")

parser.add_argument(
    "--epochs", type=int, help="Number of epochs.", required=True
)
parser.add_argument(
    "--model_type", type=int, help="one or two layer", required=True
)

parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default=".",
)
parser.add_argument(
    "--network_config",
    type=str,
    help="Path SNN network configuration.",
    required=True,
)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)

parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)
args = parser.parse_args()

params = snn.params(args.network_config)
writer = SummaryWriter(".")

# implement one layer
class OneLayerMLP(torch.nn.Module):
    def __init__(self, params, input_size,output_size):
        super(OneLayerMLP, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.fc = self.slayer.dense(input_size, output_size)

    def forward(self, spike_input):
        spike_output = self.slayer.spike(self.slayer.psp(self.fc(spike_input)))
        return spike_output

class OneLayerAE(torch.nn.Module):
    def __init__(self, params, hidden_size):
        super(OneLayerAE, self).__init__()
        self.encoder = OneLayerMLP(params, 156, hidden_size)
        self.decoder = OneLayerMLP(params, hidden_size, 156)

    def forward(self, spike_input):
        encoded_spike = self.encoder(spike_input)
        spike_output = self.decoder(encoded_spike)
        return spike_output, encoded_spike 


device = torch.device("cuda")
net = OneLayerAE(params, args.hidden_size).to(device)


error = snn.loss(params).to(device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.5)

train_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"train_80_20_{args.sample_file}.txt",
    output_size=20,
    spiking=True,
    mode='tact',
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
)
test_dataset = ViTacDataset(
    path=args.data_dir,
    sample_file=f"test_80_20_{args.sample_file}.txt",
    output_size=20,
    spiking=True,
    mode='tact',
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
)

def _train():
    correct = 0
    num_samples = 0
    net.train()
    for data, target, label in train_loader:
        data = data.to(device)
        target = target.to(device)
        output, _ = net.forward(data)
        num_samples += len(label)
        spike_loss = error.spikeTime(output, data)

        optimizer.zero_grad()
        spike_loss.backward()
        optimizer.step()

    writer.add_scalar("loss/train", spike_loss / len(train_loader), epoch)

def _test():
    correct = 0
    num_samples = 0
    net.eval()
    with torch.no_grad():
        for data, target, label in test_loader:
            data = data.to(device)
            target = target.to(device)
            output, _ = net.forward(data)
            num_samples += len(label)
            spike_loss = error.spikeTime(output, data)  # numSpikes

        writer.add_scalar("loss/test", spike_loss / len(test_loader), epoch)


def _save_model(epoch):
    log.info(f"Writing model at epoch {epoch}...")
    checkpoint_path = Path(args.checkpoint_dir) / f"weights_{epoch:03d}.pt"
    model_path = Path(args.checkpoint_dir) / f"model_{epoch:03d}.pt"
    torch.save(net.state_dict(), checkpoint_path)
    #torch.save(net, model_path)


for epoch in range(1, args.epochs + 1):
    _train()
    if epoch % 10 == 0:
        _test()
    if epoch % 100 == 0:
        _save_model(epoch)

with open("args.pkl", "wb") as f:
    pickle.dump(args, f)




