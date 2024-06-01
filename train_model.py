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
training_step = 0
validation_step = 0

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    global training_step

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

        # Standard training except we pass in y_input and tgt_mask
        pred = model(x, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 0, 2)

        # Apply log_softmax to predictions
        log_probs = F.log_softmax(pred, dim=2)
        # One-hot encode the expected outputs
        y_one_hot = torch.nn.functional.one_hot(y_expected, num_classes=pred.size(2)).float()

        loss = loss_fn(log_probs, y_one_hot)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # log to tensorboard
        correct_prediction = log_probs.argmax(dim=2) == y_expected
        sequence_accuracy = correct_prediction.all(dim=1).sum() / len(correct_prediction)
        token_accuracy = correct_prediction.sum() / np.prod(correct_prediction.shape)
        total_loss += loss.detach().item()
        writer.add_scalar("Loss/train", loss.detach().item(), global_step=training_step)
        writer.add_scalar("Accuracy per token/train", token_accuracy, global_step=training_step)
        writer.add_scalar("Accuracy per sequence/train", sequence_accuracy, global_step=training_step)
        training_step += 1

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    total_token_accuracy = 0
    total_sequence_accuracy = 0
    global validation_step

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

            # Standard training except we pass in y_input and src_mask
            pred = model(x, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 0, 2)

            # Apply log_softmax to predictions
            log_probs = F.log_softmax(pred, dim=2)
            # One-hot encode the expected outputs
            y_one_hot = torch.nn.functional.one_hot(y_expected, num_classes=pred.size(2)).float()

            loss = loss_fn(log_probs, y_one_hot)

            # log to tensorboard
            correct_prediction = log_probs.argmax(dim=2) == y_expected
            sequence_accuracy = correct_prediction.all(dim=1).sum() / len(correct_prediction)
            total_sequence_accuracy += sequence_accuracy
            token_accuracy = correct_prediction.sum() / np.prod(correct_prediction.shape)
            total_token_accuracy += token_accuracy
            total_loss += loss.detach().item()
            writer.add_scalar("Accuracy per token/validation", token_accuracy, global_step=validation_step)
            writer.add_scalar("Accuracy per sequence/validation", sequence_accuracy, global_step=validation_step)
            writer.add_scalar("Loss/validation", loss.detach().item(), global_step=validation_step)
            validation_step += 1

    return (total_loss / len(dataloader), total_token_accuracy / len(dataloader),
            total_sequence_accuracy / len(dataloader))


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list = [], []
    print("Training and validating model")

    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        validation_loss, token_accuracy, sequence_accuracy = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation token accuracy: {token_accuracy:.4f}")
        print(f"Validation sequence accuracy: {sequence_accuracy:.4f}")
        print()

    return train_loss_list, validation_loss_list


parameters = {
    "num_tokens": 146,
    "dim_model": 64,
    "num_heads": 1,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.1,
    "data_source": "geomusic_dataset_mdepth5.txt",
    "batch_size": 16,
    "lr": 0.001,
    "loss": torch.nn.KLDivLoss(reduction="batchmean"),
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

train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, parameters["epochs"])

torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
np.save(os.path.join(out_dir, "train_loss.npy"), train_loss_list)
np.save(os.path.join(out_dir, "validation_loss.npy"), validation_loss_list)

plt.figure()
plt.plot(train_loss_list, label="Train loss")
plt.plot(validation_loss_list, label="Validation loss")
plt.legend()
plt.savefig(os.path.join(out_dir, "loss_plot.png"))
writer.close()
