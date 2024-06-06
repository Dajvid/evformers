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

    parser.add_argument("--dim-model", type=int)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=1)
    parser.add_argument("--num-decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="../datasets/505_tecator-depth-0-8-145K")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tree-depth", type=int, default=8)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--fitness-ignore-pad", type=bool,
                        action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--attention-ignore-pad", type=bool,
                        action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if args.dim_model is None:
        args.dim_model = args.tree_depth * args.tree_width

    return args


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    out_dir = os.path.join("runs", start_timestamp)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)

    data = np.loadtxt(f"{args.dataset}.data")
    dict = pickle.load(open(f"{args.dataset}.dict", "rb"))

    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):]

    train_dataloader = batchify_data(train_data, batch_size=args.batch_size)
    val_dataloader = batchify_data(val_data, batch_size=args.batch_size)

    model = Transformer(
        dictionary=dict,
        dim_model=args.dim_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        tree_depth=args.tree_depth,
        tree_width=args.tree_width,
        dropout=args.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.KLDivLoss(reduction="none")

    with open(os.path.join(out_dir, "parameters.txt"), "w") as f:
        f.write(str(args))
        f.write(f"loss: {str(loss_fn)}")
        f.write(f"optimizer: {str(opt)}")

    fit(
        model=model,
        opt=opt,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        writer=writer,
        fitness_ignore_pad=args.fitness_ignore_pad,
        attention_ignore_pad=args.attention_ignore_pad
    )

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    writer.close()


if __name__ == '__main__':
    main()
