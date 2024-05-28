import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, num_tokens: int, dim_model: int, num_heads: int, num_encoder_layers: int,
                 num_decoder_layers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = dim_model

        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dropout=dropout)
        self.out = nn.Linear(dim_model, num_tokens)

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.linear.bias.data.zero_()
    #     self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        device = src.device
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        self.src_mask = mask

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        transformer_out = self.transformer(src, tgt, src_mask=mask)
        out = self.out(transformer_out)

        return out
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        #
        # src = self.encoder(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, self.src_mask)
        # return output