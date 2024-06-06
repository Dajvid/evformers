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
        positions = positions.to(device=self.p.device)
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
