import os
import pickle
import time

import torch
import numpy as np

from gpformer.data import batchify_data
from gpformer.model import Transformer
from torch.utils.tensorboard import SummaryWriter
from gpformer.training import fit

device = "cuda" if torch.cuda.is_available() else "cpu"
start_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
out_dir = os.path.join("runs", start_timestamp)
os.makedirs(out_dir, exist_ok=True)
writer = SummaryWriter(out_dir)

PAD_TOKEN = 0


parameters = {
    "dim_model": 16,
    "num_heads": 1,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.1,
    "dataset": "../datasets/505_tecator-depth-0-8-145K",
    "batch_size": 16,
    "lr": 0.001,
    "loss": torch.nn.KLDivLoss(reduction="none"),
    "epochs": 50,
}

data = np.loadtxt(f"{parameters["dataset"]}.data")
dict = pickle.load(open(f"{parameters["dataset"]}.dict", "rb"))
parameters["num_tokens"] = len(dict)

with open(os.path.join(out_dir, "parameters.txt"), "w") as f:
    f.write(str(parameters))

train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]

train_dataloader = batchify_data(train_data, batch_size=parameters["batch_size"])
val_dataloader = batchify_data(val_data, batch_size=parameters["batch_size"])

model = Transformer(parameters["num_tokens"], parameters["dim_model"], parameters["num_heads"],
                    parameters["num_encoder_layers"], parameters["num_decoder_layers"],
                    dropout=parameters["dropout"]).to(device)
opt = torch.optim.SGD(model.parameters(), lr=parameters["lr"])
loss_fn = parameters["loss"]

fit(model, opt, loss_fn, train_dataloader, val_dataloader, parameters["epochs"])

torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))

writer.close()
