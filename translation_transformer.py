"""
Language Translation with ``nn.Transformer`` and torchtext
==========================================================

This tutorial shows:
    - How to train a translation model from scratch using Transformer.
    - Use torchtext library to access  `Multi30k <http://www.statmt.org/wmt16/multimodal-task.html#task1>`__ dataset to train a German to English translation model.
"""
import os
import time

import numpy as np

######################################################################
# Seq2Seq Network using Transformer
# ---------------------------------
#
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks.
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
# Finally, the output of the Transformer model is passed through linear layer
# that gives unnormalized probabilities for each token in the target language.
#


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from torch.utils.tensorboard import SummaryWriter

from data import batchify_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


######################################################################
# During training, we need a subsequent word mask that will prevent the model from looking into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# Let's now define the parameters of our model and instantiate the same. Below, we also
# define our loss function which is the cross-entropy loss and the optimizer used for training.
#
torch.manual_seed(0)

data = np.loadtxt("geomusic_dataset_depth5-5.txt")
train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]



SRC_VOCAB_SIZE = 146
TGT_VOCAB_SIZE = 146
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
UNK_IDX, PAD_IDX = 1, 0

train_dataloader = batchify_data(train_data, batch_size=BATCH_SIZE)
val_dataloader = batchify_data(val_data, batch_size=BATCH_SIZE)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

######################################################################
# Let's define training and evaluation loop that will be called for each
# epoch.
#

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    total_sequence_accuracy = 0
    total_token_accuracy = 0

    for i, src in enumerate(train_dataloader):
        print(f"Batch {i + 1} / {len(train_dataloader)}\r", end="")
        tgt = src
        src = torch.tensor(src, dtype=torch.long, device=DEVICE)
        tgt = torch.tensor(tgt, dtype=torch.long, device=DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, None, src_padding_mask)# src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        pad_mask = tgt_input != 0
        correct_prediction = (logits.argmax(dim=2) == tgt_input) * pad_mask
        sequence_accuracy = correct_prediction.all(axis=1).sum() / len(pad_mask)
        token_accuracy = correct_prediction.sum() / (~pad_mask).sum()

        total_sequence_accuracy += sequence_accuracy.detach().item()
        total_token_accuracy += token_accuracy.detach().item()

    return (losses / len(list(train_dataloader)), total_token_accuracy / len(train_dataloader),
            total_sequence_accuracy / len(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0
    total_sequence_accuracy = 0
    total_token_accuracy = 0

    for src in val_dataloader:
        tgt = src
        src = torch.tensor(src, dtype=torch.long, device=DEVICE)
        tgt = torch.tensor(tgt, dtype=torch.long, device=DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, None, src_padding_mask) # src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        pad_mask = tgt_input != 0
        correct_prediction = (logits.argmax(dim=2) == tgt_input) * pad_mask
        sequence_accuracy = correct_prediction.all(axis=1).sum() / len(pad_mask)
        token_accuracy = correct_prediction.sum() / (~pad_mask).sum()

        total_sequence_accuracy += sequence_accuracy.detach().item()
        total_token_accuracy += token_accuracy.detach().item()

    return (losses / len(list(train_dataloader)), total_token_accuracy / len(train_dataloader),
            total_sequence_accuracy / len(train_dataloader))

######################################################################
# Now we have all the ingredients to train our model. Let's do it!
#

from timeit import default_timer as timer
NUM_EPOCHS = 18

start_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
out_dir = os.path.join("runs", start_timestamp)
os.makedirs(out_dir, exist_ok=True)
writer = SummaryWriter(out_dir)


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss, token_accuracy, sequence_accuracy = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss, token_accuracy, sequence_accuracy = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print(f"Validation token accuracy: {token_accuracy:.4f}")
    print(f"Validation sequence accuracy: {sequence_accuracy:.4f}")

    writer.add_scalar("Accuracy per token/validation", token_accuracy, global_step=epoch)
    writer.add_scalar("Accuracy per sequence/validation", sequence_accuracy, global_step=epoch)
    writer.add_scalar("Loss/validation", val_loss, global_step=epoch)
    writer.add_scalar("Loss/train", val_loss, global_step=epoch)
    writer.add_scalar("Accuracy per token/train", token_accuracy, global_step=epoch)
    writer.add_scalar("Accuracy per sequence/train", sequence_accuracy, global_step=epoch)
