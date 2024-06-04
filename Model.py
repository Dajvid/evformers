import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

from TreePosEncoding import TreePositionalEncodings


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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnedPositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        # Create a tensor of shape (max_len, d_model) for the positional embeddings
        self.positional_embeddings = nn.Embedding(max_len, d_model)

        # Initialize the positional embeddings
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize embeddings with a normal distribution
        nn.init.normal_(self.positional_embeddings.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, x):
        # Get the batch size and sequence length from the input
        batch_size, seq_len = x.size(0), x.size(1)

        # Create a tensor containing the positions [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Look up the positional embeddings for each position
        position_embeddings = self.positional_embeddings(positions)

        return position_embeddings


def generate_tree_levels_torch(depth):
    def dfs(node_depth, current_depth):
        if node_depth == current_depth:
            return []
        left_subtree = dfs(node_depth + 1, current_depth)
        right_subtree = dfs(node_depth + 1, current_depth)
        return [node_depth] + left_subtree + right_subtree

    levels = dfs(0, depth)
    return torch.tensor(levels)


class HybridPositionalEmbeddings(nn.Module):
    def __init__(self, max_len, d_model, dropout):
        super(HybridPositionalEmbeddings, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.seq_pos_embeddings = nn.Embedding(max_len, d_model)
        self.depth_embeddings = nn.Embedding(max_len, d_model)

        self.depths = generate_tree_levels_torch(int(math.log2(max_len + 1))).reshape(max_len, 1)
        self.positions = torch.arange(max_len)

        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding_depths = torch.zeros((max_len, d_model))
        pos_embeding_seq_pos = torch.zeros((max_len, d_model))

        pos_embedding_depths[:, 0::2] = torch.sin(pos * den)
        pos_embedding_depths[:, 1::2] = torch.cos(pos * den)

        pos_embeding_seq_pos[:, 0::2] = torch.sin(self.depths * den)
        pos_embeding_seq_pos[:, 1::2] = torch.cos(self.depths * den)

        pos_embedding = pos_embedding_depths.unsqueeze(-2)
        pos_embedding += pos_embeding_seq_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    def __init__(self, num_tokens: int, dim_model: int, num_heads: int, num_encoder_layers: int,
                 num_decoder_layers: int, dropout: float = 0.1, dim_feedforward: int = 2048):
        super().__init__()
        self.model_type = 'Transformer'
        self.dim_model = dim_model

        #self.positional_encoder = PositionalEncoding(dim_model, dropout)
        #self.positional_encoder = LearnedPositionalEncoding(num_tokens, dim_model)
        #self.positional_encoder = HybridPositionalEmbeddings(127, dim_model, dropout)
        self.positional_encoder = TreePositionalEncodings(emb_size=12, width=2, depth=6)

        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dropout=dropout,
                                          dim_feedforward=dim_feedforward)
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src: Tensor, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # device = src.device
        # mask = self._generate_square_subsequent_mask(len(src)).to(device)
        # self.src_mask = mask

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

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
