"""MP-STGNet with Moran-guided graph propagation and pyramid cross-scale fusion."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        if getattr(A, "is_sparse", False):
            n = x.size(2)
            out_dtype = x.dtype
            x_flat = x.permute(2, 0, 1, 3).reshape(n, -1)
            if x_flat.dtype in (torch.float16, torch.bfloat16):
                x_flat = x_flat.float()
            A_t = A.transpose(0, 1)
            if A_t.dtype != x_flat.dtype:
                A_t = A_t.to(dtype=x_flat.dtype)
            if x_flat.is_cuda:
                with torch.cuda.amp.autocast(enabled=False):
                    out = torch.sparse.mm(A_t, x_flat)
            else:
                out = torch.sparse.mm(A_t, x_flat)
            out = out.to(dtype=out_dtype)
            out = out.reshape(n, x.size(0), x.size(1), x.size(3)).permute(1, 2, 0, 3)
            return out.contiguous()
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(
        self, c_in, c_out, dropout, support_len=3, order=2, decomposed=True, support_gate=True
    ):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.dropout = dropout
        self.order = order
        self.support_len = support_len
        self.decomposed = decomposed
        self.num_terms = order * support_len + 1
        self.support_gate = support_gate
        if support_gate:
            # Initialized as identity scaling: 2 * sigmoid(0) = 1.
            self.term_gate_logits = nn.Parameter(torch.zeros(self.num_terms))
        else:
            self.register_parameter("term_gate_logits", None)

        if decomposed:
            self.mlp_terms = nn.ModuleList(
                [linear(c_in, c_out, bias=(term_idx == 0)) for term_idx in range(self.num_terms)]
            )
        else:
            self.mlp = linear(self.num_terms * c_in, c_out)

    def _scale(self, term_idx):
        if self.term_gate_logits is None:
            return 1.0
        return 2.0 * torch.sigmoid(self.term_gate_logits[term_idx])

    def forward(self, x, support):
        if self.decomposed:
            term_idx = 0
            h = self._scale(term_idx) * self.mlp_terms[term_idx](x)
            term_idx += 1
            for a in support:
                x1 = self.nconv(x, a)
                h = h + self._scale(term_idx) * self.mlp_terms[term_idx](x1)
                term_idx += 1
                for _ in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    h = h + self._scale(term_idx) * self.mlp_terms[term_idx](x2)
                    term_idx += 1
                    x1 = x2
            return F.dropout(h, self.dropout, training=self.training)

        out = []
        term_idx = 0
        out.append(self._scale(term_idx) * x)
        term_idx += 1
        for a in support:
            x1 = self.nconv(x, a)
            out.append(self._scale(term_idx) * x1)
            term_idx += 1
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(self._scale(term_idx) * x2)
                term_idx += 1
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class PyramidTileFusion(nn.Module):
    """
    OSM tile-pyramid fusion for adjacent zoom levels.

    Each coarse tile (r, c) has four deterministic children in the next level:
    (2r, 2c), (2r + 1, 2c), (2r, 2c + 1), (2r + 1, 2c + 1).
    The module performs learnable bottom-up aggregation and top-down broadcast,
    then uses a residual gate to avoid imposing a hard hierarchy constraint.
    """

    def __init__(
        self, channels, layer_shapes, use_moran_gate=True, alpha_init=0.1, mode="bidirectional"
    ):
        super().__init__()
        if layer_shapes is None or len(layer_shapes) < 2:
            raise ValueError("PyramidTileFusion requires at least two layer shapes.")
        self.channels = channels
        self.layer_shapes = [(int(h), int(w)) for h, w in layer_shapes]
        self.use_moran_gate = use_moran_gate
        valid_modes = {"bottom_up", "top_down", "bidirectional"}
        if mode not in valid_modes:
            raise ValueError(f"pyramid fusion mode must be one of {valid_modes}, got {mode}")
        self.mode = mode
        self.bottom_up_proj = nn.ModuleList()
        self.top_down_proj = nn.ModuleList()

        offsets = []
        offset = 0
        for h, w in self.layer_shapes:
            offsets.append(offset)
            offset += h * w
        self.num_nodes = offset

        for pair_idx in range(len(self.layer_shapes) - 1):
            coarse_h, coarse_w = self.layer_shapes[pair_idx]
            fine_h, fine_w = self.layer_shapes[pair_idx + 1]
            if fine_h != 2 * coarse_h or fine_w != 2 * coarse_w:
                raise ValueError(
                    f"OSM pyramid expects exact 2x shapes, got "
                    f"{self.layer_shapes[pair_idx]} -> {self.layer_shapes[pair_idx + 1]}"
                )

            coarse_local = torch.arange(coarse_h * coarse_w, dtype=torch.long).reshape(
                coarse_h, coarse_w
            )
            coarse_index = coarse_local.reshape(-1) + offsets[pair_idx]
            child_rows = torch.arange(coarse_h, dtype=torch.long).view(-1, 1, 1) * 2
            child_cols = torch.arange(coarse_w, dtype=torch.long).view(1, -1, 1) * 2
            row_delta = torch.tensor([0, 0, 1, 1], dtype=torch.long).view(1, 1, 4)
            col_delta = torch.tensor([0, 1, 0, 1], dtype=torch.long).view(1, 1, 4)
            child_local = (child_rows + row_delta) * fine_w + (child_cols + col_delta)
            child_index = child_local.reshape(-1, 4) + offsets[pair_idx + 1]

            self.register_buffer(f"coarse_index_{pair_idx}", coarse_index)
            self.register_buffer(f"child_index_{pair_idx}", child_index)
            self.bottom_up_proj.append(
                nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=False)
            )
            self.top_down_proj.append(nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=False))

        gate_in_channels = 2 * channels + (2 if use_moran_gate else 0)
        self.context_proj = nn.Conv2d(channels, channels, kernel_size=(1, 1), bias=True)
        self.gate_proj = nn.Conv2d(gate_in_channels, channels, kernel_size=(1, 1), bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x, moran=None):
        # x: [B, C, N, T]. Cross-level edges are represented by compact indices.
        context = None
        count = x.new_zeros(1, 1, x.size(2), 1)
        batch_size, channels, _, time_steps = x.shape

        for pair_idx in range(len(self.layer_shapes) - 1):
            coarse_index = getattr(self, f"coarse_index_{pair_idx}")
            child_index = getattr(self, f"child_index_{pair_idx}")
            child_flat = child_index.reshape(-1)

            coarse_x = x.index_select(2, coarse_index)
            child_x = x.index_select(2, child_flat)
            child_x = child_x.reshape(batch_size, channels, coarse_index.numel(), 4, time_steps)

            bottom_up = self.bottom_up_proj[pair_idx](child_x.mean(dim=3))
            top_down = self.top_down_proj[pair_idx](coarse_x)
            top_down = top_down.unsqueeze(3).expand(-1, -1, -1, 4, -1)
            top_down = top_down.reshape(batch_size, channels, child_flat.numel(), time_steps)

            if context is None:
                # Under AMP, the 1x1 projections are fp16 while x can still be fp32.
                # index_add requires matching dtypes, so allocate from projected features.
                context = bottom_up.new_zeros(batch_size, channels, x.size(2), time_steps)
            if self.mode in {"bottom_up", "bidirectional"}:
                context = context.index_add(2, coarse_index, bottom_up)
                count[:, :, coarse_index, :] += 1
            if self.mode in {"top_down", "bidirectional"}:
                context = context.index_add(2, child_flat, top_down)
                count[:, :, child_flat, :] += 1

        if context is None:
            return x
        context = context / count.to(dtype=context.dtype).clamp_min(1.0)
        update = torch.tanh(self.context_proj(context))
        gate_inputs = [x, context]
        if self.use_moran_gate:
            if moran is None:
                moran = x.new_zeros(x.size(0), 2, x.size(2), x.size(3))
            else:
                moran = moran[:, :, :, -x.size(3) :]
            gate_inputs.append(moran.to(dtype=context.dtype))
        gate_inputs[0] = gate_inputs[0].to(dtype=context.dtype)
        gate = torch.sigmoid(self.gate_proj(torch.cat(gate_inputs, dim=1)))
        return x + (torch.tanh(self.alpha) * gate * update).to(dtype=x.dtype)


class MPSTGNet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        use_moran=True,
        moran_mode="film",
        moran_clip=5.0,
        moran_eps=1e-6,
        layer_shapes=None,
        layer_aware_moran=True,
        moran_adj_calibration=True,
        moran_adj_strength=0.15,
        support_gate=True,
        decomposed_gcn=True,
        sparse_static_support=True,
        pyramid_fusion=True,
        pyramid_use_moran_gate=True,
        pyramid_alpha_init=0.1,
        pyramid_mode="bidirectional",
    ):
        super(MPSTGNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.use_moran = use_moran
        self.moran_mode = moran_mode
        self.moran_clip = moran_clip
        self.moran_eps = moran_eps
        self.layer_shapes = layer_shapes
        self.layer_aware_moran = layer_aware_moran
        self.moran_adj_calibration = moran_adj_calibration
        self.moran_adj_strength = moran_adj_strength
        self.support_gate = support_gate
        self.decomposed_gcn = decomposed_gcn
        self.sparse_static_support = sparse_static_support
        self.pyramid_mode = pyramid_mode
        self.pyramid_fusion = pyramid_fusion and pyramid_mode != "none"
        self.pyramid_use_moran_gate = pyramid_use_moran_gate
        self.pyramid_alpha_init = pyramid_alpha_init

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.pyramid_fusions = nn.ModuleList()

        self._init_layer_index(num_nodes, layer_shapes, device)
        self._init_moran_edges(supports, device)
        self.supports = self._prepare_supports(supports, device)

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        if self.pyramid_fusion:
            self.pyramid_fusions.extend(
                [
                    PyramidTileFusion(
                        residual_channels,
                        layer_shapes,
                        use_moran_gate=self.pyramid_use_moran_gate,
                        alpha_init=self.pyramid_alpha_init,
                        mode=self.pyramid_mode,
                    )
                    for _ in range(blocks)
                ]
            )
        if self.use_moran:
            self.moran_proj = nn.Conv2d(2, residual_channels, kernel_size=(1, 1), bias=True)
            self.moran_gate = nn.Conv2d(2, residual_channels, kernel_size=(1, 1), bias=True)
            if self.layer_aware_moran:
                num_layers = int(self.layer_id.max().item() + 1) if self.layer_id.numel() > 0 else 1
                self.layer_moran_scale = nn.Parameter(torch.ones(num_layers, residual_channels))
                self.layer_moran_bias = nn.Parameter(torch.zeros(num_layers, residual_channels))

        receptive_field = 1
        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device), requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device), requires_grad=True
                ).to(device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                            decomposed=self.decomposed_gcn,
                            support_gate=self.support_gate,
                        )
                    )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True
        )
        self.receptive_field = receptive_field

    def _init_layer_index(self, num_nodes, layer_shapes, device):
        if layer_shapes is None:
            layer_id = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            ids = []
            total = 0
            for idx, shape in enumerate(layer_shapes):
                h, w = int(shape[0]), int(shape[1])
                count = h * w
                ids.append(torch.full((count,), idx, dtype=torch.long, device=device))
                total += count
            if total != num_nodes:
                raise ValueError(
                    f"layer_shapes node count {total} does not match num_nodes {num_nodes}"
                )
            layer_id = torch.cat(ids, dim=0)
        self.register_buffer("layer_id", layer_id)

    def _prepare_supports(self, supports, device):
        if supports is None:
            return supports
        prepared = []
        for support in supports:
            support = support.detach().clone().float().to(device)
            if self.sparse_static_support:
                support = support.to_sparse().coalesce()
            prepared.append(support)
        return prepared

    def _init_moran_edges(self, supports, device):
        if (not self.use_moran) or supports is None or len(supports) == 0:
            self.register_buffer("moran_src", torch.empty(0, dtype=torch.long, device=device))
            self.register_buffer("moran_dst", torch.empty(0, dtype=torch.long, device=device))
            self.register_buffer("moran_w", torch.empty(0, dtype=torch.float32, device=device))
            self.register_buffer(
                "moran_weight_sum", torch.tensor(1.0, dtype=torch.float32, device=device)
            )
            self.register_buffer(
                "moran_dense_prior", torch.eye(1, dtype=torch.float32, device=device)
            )
            return

        adj = supports[0].detach().clone().float().to(device)
        if getattr(adj, "is_sparse", False):
            adj = adj.coalesce().to_dense()
        if adj.dim() != 2 or adj.size(0) != adj.size(1):
            raise ValueError(f"Moran adjacency must be square, got {tuple(adj.shape)}")

        adj.fill_diagonal_(0.0)
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sum.clamp_min(self.moran_eps)
        adj = torch.where(row_sum > self.moran_eps, adj, torch.zeros_like(adj))
        src, dst = torch.nonzero(adj, as_tuple=True)
        w = adj[src, dst].contiguous()

        self.register_buffer("moran_src", src.long())
        self.register_buffer("moran_dst", dst.long())
        self.register_buffer("moran_w", w.float())
        self.register_buffer("moran_weight_sum", w.sum().clamp_min(self.moran_eps).float())
        self.register_buffer("moran_dense_prior", adj.float())

    def _compute_moran_features(self, input):
        B, _, N, T = input.shape
        if self.moran_w.numel() == 0:
            return input.new_zeros(B, 2, N, T)
        if self.moran_src.max().item() >= N or self.moran_dst.max().item() >= N:
            raise ValueError(f"Moran edge index is larger than input node count {N}")

        values = input[:, :1, :, :].squeeze(1).detach()
        z = values - values.mean(dim=1, keepdim=True)
        neighbor_z = torch.zeros_like(z)
        edge_w = self.moran_w.to(device=z.device, dtype=z.dtype).view(1, -1, 1)
        src = self.moran_src.to(z.device)
        dst = self.moran_dst.to(z.device)
        messages = z.index_select(1, dst) * edge_w
        neighbor_z.index_add_(1, src, messages)

        denom = (z * z).sum(dim=1, keepdim=True).clamp_min(self.moran_eps)
        local_i = max(N - 1, 1) * z * neighbor_z / denom
        global_i = (
            (float(N) / self.moran_weight_sum.to(device=z.device, dtype=z.dtype))
            * (z * neighbor_z).sum(dim=1, keepdim=True)
            / denom
        )
        global_i = global_i.expand(-1, N, -1)

        moran = torch.stack((local_i, global_i), dim=1)
        moran = torch.nan_to_num(moran, nan=0.0, posinf=self.moran_clip, neginf=-self.moran_clip)
        return moran.clamp(min=-self.moran_clip, max=self.moran_clip)

    def _start_with_moran(self, input):
        x = self.start_conv(input)
        if not self.use_moran:
            return x, None
        moran = self._compute_moran_features(input)
        moran_add = torch.tanh(self.moran_proj(moran))
        moran_gate = torch.sigmoid(self.moran_gate(moran))
        if self.layer_aware_moran and hasattr(self, "layer_moran_scale"):
            scale = (
                self.layer_moran_scale.index_select(0, self.layer_id)
                .t()
                .view(1, -1, input.size(2), 1)
            )
            bias = (
                self.layer_moran_bias.index_select(0, self.layer_id)
                .t()
                .view(1, -1, input.size(2), 1)
            )
            moran_add = moran_add * scale + bias
        if self.moran_mode == "add":
            return x + moran_add, moran
        return x + moran_add * moran_gate, moran

    def _moran_guided_adaptive_adj(self, moran):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        if (not self.moran_adj_calibration) or moran is None or self.moran_dense_prior.numel() <= 1:
            return adp
        local_i = moran[:, 0].mean(dim=(0, 2)).detach()
        local_i = torch.nan_to_num(
            local_i, nan=0.0, posinf=self.moran_clip, neginf=-self.moran_clip
        )
        local_i = torch.sigmoid(local_i).to(device=adp.device, dtype=adp.dtype)
        node_gate = 0.5 * (local_i.view(-1, 1) + local_i.view(1, -1))
        prior = self.moran_dense_prior.to(device=adp.device, dtype=adp.dtype)
        calibrated_prior = prior * node_gate
        mixed = (1.0 - self.moran_adj_strength) * adp + self.moran_adj_strength * calibrated_prior
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(self.moran_eps)

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x_input = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x_input = input
        x, moran = self._start_with_moran(x_input)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = self._moran_guided_adaptive_adj(moran)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            # Fuse parent-child context before temporal convolution and skip accumulation.
            if self.pyramid_fusion and i % self.layers == 0:
                block_idx = i // self.layers
                x = self.pyramid_fusions[block_idx](x, moran)

            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
