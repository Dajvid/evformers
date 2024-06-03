import os
import random
import time

import torch
import numpy as np

from data import batchify_data
from Model import Transformer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
start_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
out_dir = os.path.join("runs", start_timestamp)
os.makedirs(out_dir, exist_ok=True)
writer = SummaryWriter(out_dir)

PAD_TOKEN = 0


def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    total_sequence_accuracy = 0
    total_token_accuracy = 0

    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1} / {len(dataloader)}\r", end="")
        x, y = batch, batch
        x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        src_key_padding_mask = (x == PAD_TOKEN).to(torch.bool).to(device)
        tgt_key_padding_mask = (y_input == PAD_TOKEN).to(torch.bool).to(device)

        pred = model(x, y_input, tgt_mask=tgt_mask, src_pad_mask=src_key_padding_mask,
                     tgt_pad_mask=tgt_key_padding_mask)

        # Permute pred to have batch size first again
        #log_probs = F.log_softmax(pred.permute(1, 0, 2), dim=2)
        pred = pred.permute(1, 0, 2)

        # Apply log_softmax to predictions
        log_probs = F.log_softmax(pred, dim=2)
        # One-hot encode the expected outputs
        y_one_hot = torch.nn.functional.one_hot(y_expected, num_classes=pred.size(2)).float()
        loss = loss_fn(log_probs, y_one_hot)
        # loss = loss_fn(pred, y_expected)
        pad_mask = y_expected != PAD_TOKEN
        pad_mask_unsquezed = pad_mask.unsqueeze(-1).expand_as(loss)
        masked_kl_loss = loss * (pad_mask_unsquezed.float())
        loss_sum = masked_kl_loss.sum()
        loss = loss_sum / len(pad_mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # compute statistics
        correct_prediction = (log_probs.argmax(dim=2) == y_expected) * pad_mask
        sequence_accuracy = (correct_prediction + (~pad_mask)).all(axis=1).sum() / len(pad_mask)
        token_accuracy = correct_prediction.sum() / (pad_mask).sum()

        total_sequence_accuracy += sequence_accuracy.detach().item()
        total_token_accuracy += token_accuracy.detach().item()
        total_loss += loss.detach().item()

    return (total_loss / len(dataloader), total_token_accuracy / len(dataloader),
            total_sequence_accuracy / len(dataloader))


def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    total_token_accuracy = 0
    total_sequence_accuracy = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch, batch
            x, y = torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            src_key_padding_mask = (x == PAD_TOKEN).to(torch.bool).to(device)
            tgt_key_padding_mask = (y_input == PAD_TOKEN).to(torch.bool).to(device)

            pred = model(x, y_input, tgt_mask=tgt_mask, src_pad_mask=src_key_padding_mask,
                         tgt_pad_mask=tgt_key_padding_mask)

            # Permute pred to have batch size first again
            #log_probs = F.log_softmax(pred.permute(1, 0, 2), dim=2)
            pred = pred.permute(1, 0, 2)

            # Apply log_softmax to predictions
            log_probs = F.log_softmax(pred, dim=2)
            # One-hot encode the expected outputs
            y_one_hot = torch.nn.functional.one_hot(y_expected, num_classes=pred.size(2)).float()
            loss = loss_fn(log_probs, y_one_hot)
            # loss = loss_fn(pred, y_expected)

            # compute statistics
            pad_mask = y_expected != PAD_TOKEN
            pad_mask_unsquezed = pad_mask.unsqueeze(-1).expand_as(loss)
            masked_kl_loss = loss * (pad_mask_unsquezed.float())
            loss_sum = masked_kl_loss.sum()
            loss = loss_sum / len(pad_mask)

            correct_prediction = (log_probs.argmax(dim=2) == y_expected) * pad_mask

            sequence_accuracy = (correct_prediction + (~pad_mask)).all(axis=1).sum() / len(pad_mask)
            token_accuracy = correct_prediction.sum() / (pad_mask).sum()

            total_sequence_accuracy += sequence_accuracy.detach().item()
            total_token_accuracy += token_accuracy.detach().item()
            total_loss += loss.detach().item()

    return (total_loss / len(dataloader), total_token_accuracy / len(dataloader),
            total_sequence_accuracy / len(dataloader))


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list = [], []
    print("Training and validating model")

    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        train_loss, token_accuracy, sequence_accuracy = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        validation_loss, token_accuracy, sequence_accuracy = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        # log to tensorboard
        writer.add_scalar("Accuracy per token/validation", token_accuracy, global_step=epoch)
        writer.add_scalar("Accuracy per sequence/validation", sequence_accuracy, global_step=epoch)
        writer.add_scalar("Loss/validation", validation_loss, global_step=epoch)
        writer.add_scalar("Loss/train", validation_loss, global_step=epoch)
        writer.add_scalar("Accuracy per token/train", token_accuracy, global_step=epoch)
        writer.add_scalar("Accuracy per sequence/train", sequence_accuracy, global_step=epoch)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation token accuracy: {token_accuracy:.4f}")
        print(f"Validation sequence accuracy: {sequence_accuracy:.4f}")
        print()

    return


parameters = {
    "num_tokens": 146,
    "dim_model": 64,
    "num_heads": 1,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.1,
    "data_source": "geomusic_dataset_depth5-5.txt",
    "batch_size": 16,
    "lr": 0.001,
    "loss": torch.nn.KLDivLoss(reduction="none"),
    #"loss": torch.nn.CrossEntropyLoss(ignore_index=0),
    "epochs": 50,
}
with open(os.path.join(out_dir, "parameters.txt"), "w") as f:
    f.write(str(parameters))

data = np.loadtxt(parameters["data_source"])
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
