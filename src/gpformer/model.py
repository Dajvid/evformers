import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

from .TreePosEncoding import TreePositionalEncodings

class Transformer(nn.Module):
    def __init__(self, dictionary: dict, dim_model: int, num_heads: int, num_encoder_layers: int,
                 num_decoder_layers: int, tree_depth: int, tree_width: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.dim_model = dim_model
        self.dictionary = dictionary
        num_tokens = len(dictionary)

        #self.positional_encoder = PositionalEncoding(dim_model, dropout)
        #self.positional_encoder = LearnedPositionalEncoding(num_tokens, dim_model)
        #self.positional_encoder = HybridPositionalEmbeddings(127, dim_model, dropout)
        self.positional_encoder = TreePositionalEncodings(emb_size=dim_model, width=tree_width, depth=tree_depth)

        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dropout=dropout,
                                          dim_feedforward=dim_feedforward, batch_first=True)
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src: Tensor, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None) -> Tensor:
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoder(src, mode="src")
        tgt = self.positional_encoder(tgt, mode="tgt")

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        mask = mask.masked_fill(mask ==  float('-inf'), 1)

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask.to(torch.bool)
