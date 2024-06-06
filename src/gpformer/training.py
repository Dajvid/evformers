import os
import time

import torch

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

PAD_TOKEN = 0


def check_args_compatibility(mode, opt):
    if mode == "train" and opt is None:
        raise ValueError("Optimizer is required in training mode")
    if mode == "eval" and opt is not None:
        raise ValueError("Optimizer is not required in evaluation mode")
    if mode not in ["train", "eval"]:
        raise ValueError("Mode must be either 'train' or 'eval'")


def run_epoch(model, loss_fn, dataloader, mode, opt=None,
              fitness_ignore_pad=True, attention_ignore_pad=True):
    check_args_compatibility(mode, opt)
    model.train() if mode == "train" else model.eval()

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
        src_key_padding_mask = (x == PAD_TOKEN).to(torch.bool).to(device) if attention_ignore_pad else None
        tgt_key_padding_mask = (y_input == PAD_TOKEN).to(torch.bool).to(device) if attention_ignore_pad else None

        pred = model(x, y_input, tgt_mask=tgt_mask, src_pad_mask=src_key_padding_mask,
                     tgt_pad_mask=tgt_key_padding_mask)

        # Apply log_softmax to predictions
        log_probs = F.log_softmax(pred, dim=2)
        # One-hot encode the expected outputs
        y_one_hot = torch.nn.functional.one_hot(y_expected, num_classes=pred.size(2)).float()
        loss = loss_fn(log_probs, y_one_hot)
        pad_mask = y_expected != PAD_TOKEN
        pad_mask_unsquezed = pad_mask.unsqueeze(-1).expand_as(loss)
        masked_kl_loss = loss * (pad_mask_unsquezed.float())
        loss_sum = masked_kl_loss.sum() if fitness_ignore_pad else loss.sum()
        loss = loss_sum / len(pad_mask)

        if mode == "train":
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


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, writer=None):
    print("Training and validating model")

    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1} / {epochs}", "-" * 25)
        train_loss, train_token_accuracy, train_sequence_accuracy = run_epoch(model, loss_fn, train_dataloader,
                                                                              opt=opt, mode="train")
        val_loss, val_token_accuracy, val_sequence_accuracy = run_epoch(model, loss_fn, val_dataloader, mode="eval")

        # log to tensorboard
        if writer:
            writer.add_scalar("Accuracy per token/validation", val_token_accuracy, global_step=epoch)
            writer.add_scalar("Accuracy per sequence/validation", val_sequence_accuracy, global_step=epoch)
            writer.add_scalar("Loss/validation", val_loss, global_step=epoch)
            writer.add_scalar("Loss/train", train_loss, global_step=epoch)
            writer.add_scalar("Accuracy per token/train", train_token_accuracy, global_step=epoch)
            writer.add_scalar("Accuracy per sequence/train", train_sequence_accuracy, global_step=epoch)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation token accuracy: {val_token_accuracy:.4f}")
        print(f"Validation sequence accuracy: {val_sequence_accuracy:.4f}")
        print()

    return
