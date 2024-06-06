import os
import pickle
import time
import torch
import argparse
import numpy as np

from gpformer.data import batchify_data
from gpformer.model import Transformer
from torch.utils.tensorboard import SummaryWriter
from gpformer.training import fit


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument("--dim-model", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-encoder-layers", type=int, default=1)
    parser.add_argument("--num-decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="../datasets/505_tecator-depth-0-8-145K")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", type=str, default="KLDivLoss")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    out_dir = os.path.join("runs", start_timestamp)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)

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

    model = Transformer(dict,
                        dim_model=parameters["dim_model"],
                        num_heads=parameters["num_heads"],
                        num_encoder_layers=parameters["num_encoder_layers"],
                        num_decoder_layers=parameters["num_decoder_layers"],
                        tree_depth=8,
                        tree_width=2,
                        dropout=parameters["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])
    loss_fn = parameters["loss"]

    fit(model, opt, loss_fn, train_dataloader, val_dataloader, parameters["epochs"],
        writer=writer, idx_pad=dict["PAD"])

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))

    writer.close()


if __name__ == '__main__':
    main()
