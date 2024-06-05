import math
import torch


class TreePositionalEncodings(torch.nn.Module):
    def __init__(self, emb_size, width, depth):
        super(TreePositionalEncodings, self).__init__()
        self.depth = depth
        self.width = width
        self.d_tree_param = emb_size // depth // width
        self.d_pos = emb_size
        self.p = torch.nn.Parameter(torch.ones(self.d_tree_param, dtype=torch.float32), requires_grad=True)
        self.init_weights()
        self.positions = self.calculate_positions(depth, width)

    def calculate_positions(self, max_depth, degree):

        def dfs_step(tree_depth, current_depth, step_direction, current_path, paths):
            if tree_depth == current_depth:
                return
            if step_direction == "left":
                current_path = [1] + [0] + current_path[:-2]
            elif step_direction == "right":
                current_path = [0] + [1] + current_path[:-2]
            paths.append(current_path)

            dfs_step(tree_depth, current_depth + 1, "left", current_path, paths)
            dfs_step(tree_depth, current_depth + 1, "right", current_path, paths)

        paths = []
        current_path = [0 for _ in range(degree * max_depth)]
        paths.append(current_path)
        dfs_step(max_depth, 0, "left", current_path, paths)
        dfs_step(max_depth, 0, "right", current_path, paths)

        return torch.tensor(paths, dtype=torch.float32, device=self.p.device)


    def init_weights(self):
        self.p.data.uniform_(0.7, 0.999)

    def build_weights(self):
        d_tree_param = self.d_tree_param
        tree_params = torch.tanh(self.p)
        tiled_tree_params = tree_params.reshape((1, 1, -1)).repeat(self.depth, self.width, 1)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device=self.p.device) \
            .reshape(-1, 1, 1).repeat(1, self.width, d_tree_param)
        tree_norm = torch.sqrt((1 - tree_params ** 2) * self.d_pos / 2)
        tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm) \
            .reshape(self.depth * self.width, d_tree_param).to(device=self.p.device)
        return tree_weights

    def treeify_positions(self, positions, tree_weights):
        treeified = positions.unsqueeze(-1) * tree_weights
        shape = treeified.shape[:-2] + (self.d_pos,)
        return treeified.reshape(shape)

    def forward(self, token_embedding: torch.Tensor, mode):
        """
            positions: Tensor [bs, n, width * depth]
            returns: Tensor [bs, n, width * depth * n_features]
        """
        tree_weights = self.build_weights()

        positions = self.treeify_positions(self.positions, tree_weights)

        if mode == "tgt":
            positions = positions[:-1, :]

        return token_embedding + positions


# class TreePositionalEncodings(torch.nn.Module):
#     # Novel positional encodings to enable tree-based transformers
#     # https://papers.nips.cc/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf
#     def __init__(self, depth, degree, n_feat, d_model):
#         """
#             depth: max tree depth
#             degree: max num children
#             n_feat: number of features
#             d_model: size of model embeddings
#         """
#         super(TreePositionalEncodings, self).__init__()
#         self.depth = depth
#         self.width = degree
#         self.d_pos = n_feat * depth * degree
#         self.d_model = d_model
#         self.d_tree_param = self.d_pos // (self.depth * self.width)
#         self.p = torch.nn.Parameter(torch.ones(self.d_tree_param, dtype=torch.float32), requires_grad=True)
#         self.init_weights()
#
#     def init_weights(self):
#         self.p.data.uniform_(0.7, 0.999)
#
#     def build_weights(self):
#         d_tree_param = self.d_tree_param
#         tree_params = torch.tanh(self.p)
#         tiled_tree_params = tree_params.reshape((1, 1, -1)).repeat(self.depth, self.width, 1)
#         tiled_depths = torch.arange(self.depth, dtype=torch.float32, device=self.p.device) \
#             .reshape(-1, 1, 1).repeat(1, self.width, d_tree_param)
#         tree_norm = torch.sqrt((1 - torch.square(tree_params)) * self.d_model / 2)
#         tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm) \
#             .reshape(self.depth * self.width, d_tree_param)
#         return tree_weights
#
#     def treeify_positions(self, positions, tree_weights):
#         treeified = positions.unsqueeze(-1) * tree_weights
#         shape = treeified.shape
#         shape = shape[:-2] + (self.d_pos,)
#         treeified = torch.reshape(treeified, shape)
#         return treeified
#
#     def forward(self, positions):
#         """
#             positions: Tensor [bs, n, width * depth]
#             returns: Tensor [bs, n, width * depth * n_features]
#         """
#         tree_weights = self.build_weights()
#         positions = self.treeify_positions(positions, tree_weights)
#         return positions


# class Meta:
#     def __init__(self, args):
#         assert args.d_embed % (args.max_depth * args.max_width) == 0
#         self.max_depth = args.max_depth
#         self.max_width = args.max_width
#         self.num_feat = args.d_embed // (args.max_depth * args.max_width)
#
# def build_tree_pos_enc_meta(args):
#     return Meta(args) if args.tree_pos_enc else None
#
#
# class PositionalEncodings(torch.nn.Module):
#     def __init__(self, n_ctx, n_embd, use_sin_pos_enc=False, use_pos_embed=False, tree_pos_enc_meta=None, path_lstm=None, embed_dropout=0.1):
#         super(PositionalEncodings, self).__init__()
#         self.use_pos_enc = (tree_pos_enc_meta is not None) or use_sin_pos_enc or (path_lstm is not None)
#         self.tree_pos_enc = None
#         if tree_pos_enc_meta is not None:
#             self.tree_pos_enc = TreePositionalEncodings(
#                 depth=tree_pos_enc_meta.max_depth,
#                 degree=tree_pos_enc_meta.max_width,
#                 n_feat=tree_pos_enc_meta.num_feat,
#                 d_model=n_embd
#             )
#         assert not (use_sin_pos_enc and use_pos_embed), "use either encodings or embeddings (or none)"
#         self.pos = None
#         if use_sin_pos_enc:
#             self.pos = SinusoidPositionalEncodings(0.0, n_embd, max_len=n_ctx)
#         elif use_pos_embed:
#             self.pos = PositionalEmbeddings(n_ctx, n_embd)
#         self.emb_dropout = torch.nn.Dropout(embed_dropout)
#
#     def forward(self, hidden_states, paths=None, positions=None):
#         if self.use_pos_enc:
#             hidden_states = hidden_states \
#                 * math.sqrt(hidden_states.size(-1))
#         if self.tree_pos_enc is not None:  # tree positional encodings
#             hidden_states += self.tree_pos_enc(positions)
#         if self.pos is not None:  # positional encodings/embeddings
#             hidden_states += self.pos(hidden_states)
#         hidden_states = self.emb_dropout(hidden_states)
#         return hidden_states
