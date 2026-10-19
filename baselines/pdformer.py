"""PDFormer baseline adapted for multilevel OSM tile-access forecasting."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.util import load_city

from scripts.metrics import (
    finite_mean,
    legacy_metric_all,
    legacy_metric_nz,
    legacy_metric_triplet,
    legacy_pearson,
)

import os


import gc


import inspect


import math


import random


import re


from contextlib import contextmanager


from typing import List, Tuple


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from scipy import sparse


from scipy.sparse.linalg import eigsh


from torch.utils.checkpoint import checkpoint


from torch.utils.data import DataLoader, Dataset


import matplotlib


import matplotlib.pyplot as plt


DATA_ROOT = REPO_ROOT / "data"

RESULT_ROOT = REPO_ROOT / "runs/baselines/results"

MODEL_ROOT = REPO_ROOT / "runs/baselines/checkpoints"

CACHE_ROOT = REPO_ROOT / "runs/baselines/cache"

PLOT_ROOT = REPO_ROOT / "runs/baselines/figures"

MODEL_NAME = "PDFormer_Baseline_IntraLayerOnly"


RUN_CITIES = ["Rome", "Milan", "Vienna", "SanFrancisco", "Seoul"]

SEEDS_TO_RUN = [42, 76, 99]


TRAIN_SPLIT_RATIO = 0.6


VALIDATION_SPLIT_RATIO = 0.2


INPUT_STEPS = 28


OUTPUT_STEPS = 7


BATCH_SIZE = 16


GRAD_ACCUM_STEPS = 16


NUM_EPOCHS = 300


PATIENCE = 20


MIN_DELTA = 1e-4


LEARNING_RATE = 1e-3


MIN_LEARNING_RATE = 1e-4


WARMUP_EPOCHS = 5


WEIGHT_DECAY = 0.05


MAX_GRAD_NORM = 5.0


SKIP_COMPLETED = True


EMBED_DIM = 64


SKIP_DIM = 256


ENC_DEPTH = 6


GEO_NUM_HEADS = 2


SEM_NUM_HEADS = 2


TEMP_NUM_HEADS = 4


MLP_RATIO = 4.0


DROP = 0.0


ATTN_DROP = 0.0


DROP_PATH = 0.3


S_ATTN_SIZE = 3


LAPE_DIM = 8


GEO_HOP_THRESHOLD = 3


SEMANTIC_K = 5


SEMANTIC_PROFILE_BINS = 28


PATTERN_CLUSTERS = 16


PATTERN_SAMPLE_LIMIT = 100_000


PATTERN_KMEANS_ITERS = 20


SPATIAL_CHUNK_SIZE = 128


PREPROCESS_SEED = 2026


USE_ACTIVATION_CHECKPOINT = True


USE_AMP = True


DEVICE = torch.device("cpu")  # Resolved explicitly by run_baseline.py.


def set_seed(seed: int) -> None:
    """每个城市开始前调用，锁定 Python/NumPy/PyTorch 随机状态。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    print(f"Seed fixed before city run: {seed}")


@contextmanager
def autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type="cuda", enabled=enabled):
            yield
    else:
        with torch.cuda.amp.autocast(enabled=enabled):
            yield


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _layer_sort_key(path: Path) -> int:
    nums = re.findall(r"\d+", path.name)
    if not nums:
        raise ValueError(f"Layer file has no numeric layer id: {path}")
    return int(nums[0])


def load_city_layers(city, num_layers=4):
    if not 1 <= num_layers <= 4:
        raise ValueError("num_layers must be between 1 and 4")
    arrays, _ = load_city(DATA_ROOT, city)
    return [14, 15, 16, 17][:num_layers], arrays[:num_layers]


def normalize_data_levels(data_list, train_time_end, epsilon=1e-5):
    normalized, params = [], []
    for data in data_list:
        train_slice = data[:train_time_end]
        mean = train_slice.mean()
        std = train_slice.std()
        if float(std) < epsilon:
            std = torch.tensor(epsilon, dtype=data.dtype)
        normalized.append((data - mean) / std)
        params.append({"mean": mean.cpu(), "std": std.cpu()})
    return normalized, params


class MultiLayerPDFormerDataset(Dataset):
    def __init__(self, data_list, input_steps, output_steps, time_start=0, time_end=None):
        self.data_list = data_list
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        total = int(data_list[0].shape[0])
        time_end = total if time_end is None else int(time_end)
        max_start = time_end - self.input_steps - self.output_steps
        if max_start < time_start:
            raise ValueError(f"Time range [{time_start}, {time_end}) is too short")
        self.valid_indices = list(range(int(time_start), max_start + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start = self.valid_indices[index]
        input_end = start + self.input_steps
        target_end = input_end + self.output_steps
        inputs = [data[start:input_end] for data in self.data_list]
        targets = [data[input_end:target_end] for data in self.data_list]
        # 数据为逐日序列；绝对起始星期未知，但固定模 7 标签仍保留周周期分组。
        input_dow = torch.arange(start, input_end, dtype=torch.long) % 7
        return inputs, targets, input_dow


def make_dataloaders(normalized_data_levels, total_timesteps):
    train_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    val_end = int(total_timesteps * (TRAIN_SPLIT_RATIO + VALIDATION_SPLIT_RATIO))
    train_ds = MultiLayerPDFormerDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, 0, train_end
    )
    val_ds = MultiLayerPDFormerDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, train_end, val_end
    )
    test_ds = MultiLayerPDFormerDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, val_end, total_timesteps
    )
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return (
        DataLoader(train_ds, **kwargs),
        DataLoader(val_ds, **kwargs),
        DataLoader(test_ds, **kwargs),
        train_end,
        val_end,
    )


def layers_to_node_tensor(layers: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """[(B,T,H,W,1), ...] -> (B,T,N,1)."""
    flattened = []
    batch_size = time_steps = None
    for index, layer in enumerate(layers):
        layer = layer.to(device, non_blocking=True)
        if layer.ndim != 5:
            raise ValueError(f"Layer {index} should be (B,T,H,W,C), got {tuple(layer.shape)}")
        b, t, h, w, c = layer.shape
        if c != 1:
            raise ValueError(f"PDFormer baseline expects one raw channel, got C={c}")
        if index == 0:
            batch_size, time_steps = b, t
        elif (b, t) != (batch_size, time_steps):
            raise ValueError("Batch/time mismatch between zoom layers")
        flattened.append(layer.contiguous().view(b, t, h * w, 1))
    return torch.cat(flattened, dim=2)


def split_node_tensor(x: torch.Tensor, shapes: List[Tuple[int, int]]):
    """(B,T,N,1) -> [(B,T,H,W,1), ...]."""
    expected = sum(h * w for h, w in shapes)
    if x.ndim != 4 or x.shape[2] != expected or x.shape[-1] != 1:
        raise ValueError(f"Unexpected model tensor shape {tuple(x.shape)}, expected N={expected}")
    output, pointer = [], 0
    for height, width in shapes:
        nodes = height * width
        output.append(
            x[:, :, pointer : pointer + nodes].view(x.shape[0], x.shape[1], height, width, 1)
        )
        pointer += nodes
    return output


def build_layer_ids(shapes):
    return np.concatenate(
        [
            np.full(height * width, layer_index, dtype=np.int64)
            for layer_index, (height, width) in enumerate(shapes)
        ]
    )


def build_geo_neighbors(shapes, hop_threshold=3):
    """构造同层 Chebyshev shortest-hop 邻域及有效掩码。"""
    radius = int(hop_threshold) - 1
    if radius < 0:
        raise ValueError("hop_threshold must be >= 1")
    max_neighbors = (2 * radius + 1) ** 2
    total_nodes = sum(height * width for height, width in shapes)
    neighbors = np.zeros((total_nodes, max_neighbors), dtype=np.int64)
    valid = np.zeros((total_nodes, max_neighbors), dtype=np.bool_)
    offset = 0
    for height, width in shapes:
        for row in range(height):
            for col in range(width):
                node = offset + row * width + col
                items = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr, cc = row + dr, col + dc
                        if 0 <= rr < height and 0 <= cc < width:
                            items.append(offset + rr * width + cc)
                neighbors[node, : len(items)] = items
                neighbors[node, len(items) :] = node
                valid[node, : len(items)] = True
        offset += height * width
    return neighbors, valid


def compress_profiles(series_tn, bins):
    """将训练序列压缩为固定长度轮廓，降低全节点近邻检索成本。"""
    time_steps, num_nodes = series_tn.shape
    edges = np.linspace(0, time_steps, bins + 1, dtype=np.int64)
    features = []
    for index in range(bins):
        start, end = int(edges[index]), int(edges[index + 1])
        end = max(end, start + 1)
        features.append(series_tn[start:end].mean(axis=0))
    profile = np.stack(features, axis=1).astype(np.float32)
    profile -= profile.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(profile, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return profile / norm


def topk_profile_neighbors(profile, k, chunk_size=256):
    num_nodes = profile.shape[0]
    use_k = min(int(k), num_nodes)
    result = np.empty((num_nodes, int(k)), dtype=np.int64)
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        similarity = profile[start:end] @ profile.T
        candidate = np.argpartition(-similarity, kth=use_k - 1, axis=1)[:, :use_k]
        candidate_score = np.take_along_axis(similarity, candidate, axis=1)
        for local_row in range(end - start):
            order = np.lexsort((candidate[local_row], -candidate_score[local_row]))
            selected = candidate[local_row, order]
            result[start + local_row, :use_k] = selected
            if use_k < k:
                result[start + local_row, use_k:] = start + local_row
    return result


def build_semantic_neighbors(normalized_levels, train_end, k, bins):
    all_neighbors = []
    offset = 0
    for data in normalized_levels:
        series = data[:train_end, :, :, 0].reshape(train_end, -1).cpu().numpy()
        profile = compress_profiles(series, bins)
        local = topk_profile_neighbors(profile, k)
        all_neighbors.append(local + offset)
        offset += local.shape[0]
    return np.concatenate(all_neighbors, axis=0)


def grid_adjacency(height, width):
    rows, cols = [], []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for row in range(height):
        for col in range(width):
            source = row * width + col
            for dr, dc in directions:
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    rows.append(source)
                    cols.append(rr * width + cc)
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.coo_matrix((data, (rows, cols)), shape=(height * width, height * width)).tocsr()


def canonicalize_eigenvector_signs(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return vectors


def layer_laplacian_pe(height, width, dimensions):
    adjacency = grid_adjacency(height, width)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree[degree <= 0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized_adj = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    laplacian = sparse.eye(adjacency.shape[0], format="csr") - normalized_adj
    num_nodes = adjacency.shape[0]
    requested = min(dimensions + 1, max(num_nodes - 1, 1))
    if num_nodes <= dimensions + 2:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        order = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, order]
    else:
        eigenvalues, eigenvectors = eigsh(laplacian, k=requested, which="SM", tol=1e-5)
        order = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, order]
    usable = canonicalize_eigenvector_signs(eigenvectors[:, 1 : dimensions + 1])
    output = np.zeros((num_nodes, dimensions), dtype=np.float32)
    output[:, : usable.shape[1]] = usable.astype(np.float32)
    return output


def build_laplacian_pe(shapes, dimensions):
    return np.concatenate(
        [layer_laplacian_pe(height, width, dimensions) for height, width in shapes], axis=0
    )


def deterministic_kmeans(samples, clusters, iterations, seed):
    rng = np.random.default_rng(seed)
    if samples.shape[0] < clusters:
        raise ValueError("Not enough pattern samples for requested clusters")
    centers = np.empty((clusters, samples.shape[1]), dtype=np.float32)
    centers[0] = samples[rng.integers(samples.shape[0])]
    min_distance = np.sum((samples - centers[0]) ** 2, axis=1)
    for index in range(1, clusters):
        total = float(min_distance.sum())
        if total <= 1e-12:
            centers[index] = samples[index % samples.shape[0]]
        else:
            selected = rng.choice(samples.shape[0], p=min_distance / total)
            centers[index] = samples[selected]
        distance = np.sum((samples - centers[index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)

    for _ in range(iterations):
        distance = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = distance.argmin(axis=1)
        updated = centers.copy()
        for index in range(clusters):
            members = samples[labels == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centers, atol=1e-5, rtol=0):
            centers = updated
            break
        centers = updated
    return centers


def build_pattern_keys(normalized_levels, train_end, window, clusters, sample_limit, iterations):
    series = np.concatenate(
        [
            data[:train_end, :, :, 0].reshape(train_end, -1).cpu().numpy()
            for data in normalized_levels
        ],
        axis=1,
    ).astype(np.float32)
    candidate_times = train_end - window + 1
    if candidate_times <= 0:
        raise ValueError("Training split is shorter than the pattern window")
    num_nodes = series.shape[1]
    total_candidates = candidate_times * num_nodes
    rng = np.random.default_rng(PREPROCESS_SEED)
    sample_count = min(total_candidates, sample_limit)
    ids = rng.choice(total_candidates, size=sample_count, replace=False)
    times = ids // num_nodes
    nodes = ids % num_nodes
    windows = np.stack([series[times + offset, nodes] for offset in range(window)], axis=1)
    centers = deterministic_kmeans(windows, clusters, iterations, PREPROCESS_SEED)
    return centers[:, :, None].astype(np.float32)


def cache_path_for_city(city, train_end, shapes):
    shape_tag = "-".join(f"{height}x{width}" for height, width in shapes)
    return CACHE_ROOT / (
        f"{city}_train{train_end}_{shape_tag}_geo{GEO_HOP_THRESHOLD}_sem{SEMANTIC_K}_"
        f"bins{SEMANTIC_PROFILE_BINS}_lape{LAPE_DIM}_patterns{PATTERN_CLUSTERS}.npz"
    )


def prepare_city_static_features(city, normalized_levels, train_end, shapes):
    cache_path = cache_path_for_city(city, train_end, shapes)
    if cache_path.exists():
        cached = np.load(cache_path)
        result = {key: cached[key] for key in cached.files}
        expected_nodes = sum(height * width for height, width in shapes)
        if result["geo_neighbors"].shape[0] != expected_nodes:
            raise RuntimeError(f"Invalid cache node count: {cache_path}")
        print(f"Loaded PDFormer preprocessing cache: {cache_path}")
        return result

    print("Building training-only PDFormer static features...")
    geo_neighbors, geo_valid = build_geo_neighbors(shapes, GEO_HOP_THRESHOLD)
    semantic_neighbors = build_semantic_neighbors(
        normalized_levels, train_end, SEMANTIC_K, SEMANTIC_PROFILE_BINS
    )
    lap_pe = build_laplacian_pe(shapes, LAPE_DIM)
    pattern_keys = build_pattern_keys(
        normalized_levels,
        train_end,
        S_ATTN_SIZE,
        PATTERN_CLUSTERS,
        PATTERN_SAMPLE_LIMIT,
        PATTERN_KMEANS_ITERS,
    )
    layer_ids = build_layer_ids(shapes)
    np.savez_compressed(
        cache_path,
        geo_neighbors=geo_neighbors,
        geo_valid=geo_valid,
        semantic_neighbors=semantic_neighbors,
        lap_pe=lap_pe,
        pattern_keys=pattern_keys,
        layer_ids=layer_ids,
    )
    print(f"Saved PDFormer preprocessing cache: {cache_path}")
    return {
        "geo_neighbors": geo_neighbors,
        "geo_valid": geo_valid,
        "semantic_neighbors": semantic_neighbors,
        "lap_pe": lap_pe,
        "pattern_keys": pattern_keys,
        "layer_ids": layer_ids,
    }


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        return drop_path(x, self.probability, self.training)


class PDFormerDataEmbedding(nn.Module):
    def __init__(self, embed_dim, lape_dim, max_time=512, drop=0.0):
        super().__init__()
        self.value_embedding = nn.Linear(1, embed_dim)
        self.day_embedding = nn.Embedding(7, embed_dim)
        self.lape_embedding = nn.Linear(lape_dim, embed_dim)
        position = torch.arange(max_time, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )
        encoding = torch.zeros(max_time, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("position_encoding", encoding, persistent=False)
        self.dropout = nn.Dropout(drop)

    def forward(self, values, day_of_week, lap_pe):
        batch, time_steps, nodes, _ = values.shape
        x = self.value_embedding(values)
        x = x + self.position_encoding[:time_steps].view(1, time_steps, 1, -1)
        x = x + self.day_embedding(day_of_week).unsqueeze(2)
        x = x + self.lape_embedding(lap_pe).view(1, 1, nodes, -1)
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class SparsePDFormerAttention(nn.Module):
    def __init__(
        self,
        dim,
        geo_heads,
        semantic_heads,
        temporal_heads,
        pattern_window,
        spatial_chunk_size,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        total_heads = geo_heads + semantic_heads + temporal_heads
        if dim % total_heads != 0:
            raise ValueError(f"embed_dim={dim} must be divisible by total heads={total_heads}")
        self.dim = dim
        self.geo_heads = geo_heads
        self.semantic_heads = semantic_heads
        self.temporal_heads = temporal_heads
        self.head_dim = dim // total_heads
        self.geo_dim = geo_heads * self.head_dim
        self.semantic_dim = semantic_heads * self.head_dim
        self.temporal_dim = temporal_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.spatial_chunk_size = int(spatial_chunk_size)

        self.temporal_q = nn.Linear(dim, self.temporal_dim, bias=qkv_bias)
        self.temporal_k = nn.Linear(dim, self.temporal_dim, bias=qkv_bias)
        self.temporal_v = nn.Linear(dim, self.temporal_dim, bias=qkv_bias)
        self.geo_q = nn.Linear(dim, self.geo_dim, bias=qkv_bias)
        self.geo_k = nn.Linear(dim, self.geo_dim, bias=qkv_bias)
        self.geo_v = nn.Linear(dim, self.geo_dim, bias=qkv_bias)
        self.semantic_q = nn.Linear(dim, self.semantic_dim, bias=qkv_bias)
        self.semantic_k = nn.Linear(dim, self.semantic_dim, bias=qkv_bias)
        self.semantic_v = nn.Linear(dim, self.semantic_dim, bias=qkv_bias)

        self.pattern_embedding = nn.Linear(pattern_window, dim)
        self.pattern_q = nn.Linear(dim, self.geo_dim)
        self.pattern_k = nn.Linear(dim, self.geo_dim)
        self.pattern_v = nn.Linear(dim, self.geo_dim)

        self.temporal_attn_drop = nn.Dropout(attn_drop)
        self.geo_attn_drop = nn.Dropout(attn_drop)
        self.semantic_attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _temporal_attention(self, x):
        batch, time_steps, nodes, _ = x.shape
        q = self.temporal_q(x).view(batch, time_steps, nodes, self.temporal_heads, self.head_dim)
        k = self.temporal_k(x).view(batch, time_steps, nodes, self.temporal_heads, self.head_dim)
        v = self.temporal_v(x).view(batch, time_steps, nodes, self.temporal_heads, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)
        v = v.permute(0, 2, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.temporal_attn_drop(attention.softmax(dim=-1))
        output = attention @ v
        return output.permute(0, 3, 1, 2, 4).reshape(batch, time_steps, nodes, self.temporal_dim)

    def _sparse_spatial_attention(self, q, k, v, neighbors, valid_mask, dropout):
        batch, time_steps, nodes, heads, head_dim = q.shape
        chunks = []
        for start in range(0, nodes, self.spatial_chunk_size):
            end = min(start + self.spatial_chunk_size, nodes)
            index = neighbors[start:end]
            q_chunk = q[:, :, start:end]
            k_neighbors = k[:, :, index]
            v_neighbors = v[:, :, index]
            # score: (B,T,C,H,K)
            score = (q_chunk.unsqueeze(3) * k_neighbors).sum(dim=-1)
            score = score.permute(0, 1, 2, 4, 3) * self.scale
            if valid_mask is not None:
                mask = valid_mask[start:end].view(1, 1, end - start, 1, -1)
                score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
            weights = dropout(score.softmax(dim=-1))
            weights = weights.permute(0, 1, 2, 4, 3).unsqueeze(-1)
            chunks.append((weights * v_neighbors).sum(dim=3))
        return torch.cat(chunks, dim=2).reshape(batch, time_steps, nodes, heads * head_dim)

    def forward(
        self,
        x,
        x_patterns,
        pattern_keys,
        geo_neighbors,
        geo_valid,
        semantic_neighbors,
    ):
        batch, time_steps, nodes, _ = x.shape
        temporal_x = self._temporal_attention(x)

        embedded_patterns = self.pattern_embedding(x_patterns)
        embedded_keys = self.pattern_embedding(pattern_keys[..., 0])
        pattern_q = self.pattern_q(embedded_patterns)
        pattern_k = self.pattern_k(embedded_keys)
        pattern_v = self.pattern_v(embedded_keys)
        pattern_attention = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
        pattern_context = pattern_attention.softmax(dim=-1) @ pattern_v

        geo_q = self.geo_q(x).view(batch, time_steps, nodes, self.geo_heads, self.head_dim)
        geo_k = (self.geo_k(x) + pattern_context).view(
            batch, time_steps, nodes, self.geo_heads, self.head_dim
        )
        geo_v = self.geo_v(x).view(batch, time_steps, nodes, self.geo_heads, self.head_dim)
        geo_x = self._sparse_spatial_attention(
            geo_q, geo_k, geo_v, geo_neighbors, geo_valid, self.geo_attn_drop
        )

        sem_q = self.semantic_q(x).view(
            batch, time_steps, nodes, self.semantic_heads, self.head_dim
        )
        sem_k = self.semantic_k(x).view(
            batch, time_steps, nodes, self.semantic_heads, self.head_dim
        )
        sem_v = self.semantic_v(x).view(
            batch, time_steps, nodes, self.semantic_heads, self.head_dim
        )
        semantic_x = self._sparse_spatial_attention(
            sem_q, sem_k, sem_v, semantic_neighbors, None, self.semantic_attn_drop
        )

        output = self.proj(torch.cat([temporal_x, geo_x, semantic_x], dim=-1))
        return self.proj_drop(output)


class PDFormerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        geo_heads,
        semantic_heads,
        temporal_heads,
        pattern_window,
        spatial_chunk_size,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_probability=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = SparsePDFormerAttention(
            dim,
            geo_heads,
            semantic_heads,
            temporal_heads,
            pattern_window,
            spatial_chunk_size,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_probability)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_neighbors, geo_valid, semantic_neighbors):
        attention_output = self.attention(
            self.norm1(x),
            x_patterns,
            pattern_keys,
            geo_neighbors,
            geo_valid,
            semantic_neighbors,
        )
        x = x + self.drop_path(attention_output)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ScalablePDFormer(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_steps,
        output_steps,
        geo_neighbors,
        geo_valid,
        semantic_neighbors,
        lap_pe,
        pattern_keys,
        embed_dim=64,
        skip_dim=256,
        enc_depth=6,
        geo_heads=2,
        semantic_heads=2,
        temporal_heads=4,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_probability=0.3,
        pattern_window=3,
        spatial_chunk_size=128,
        use_activation_checkpoint=True,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.pattern_window = int(pattern_window)
        self.use_activation_checkpoint = bool(use_activation_checkpoint)
        self.checkpoint_has_use_reentrant = (
            "use_reentrant" in inspect.signature(checkpoint).parameters
        )

        self.register_buffer("geo_neighbors", torch.as_tensor(geo_neighbors, dtype=torch.long))
        self.register_buffer("geo_valid", torch.as_tensor(geo_valid, dtype=torch.bool))
        self.register_buffer(
            "semantic_neighbors", torch.as_tensor(semantic_neighbors, dtype=torch.long)
        )
        self.register_buffer("lap_pe", torch.as_tensor(lap_pe, dtype=torch.float32))
        self.register_buffer("pattern_keys", torch.as_tensor(pattern_keys, dtype=torch.float32))

        self.embedding = PDFormerDataEmbedding(embed_dim, lap_pe.shape[1], drop=drop)
        drop_rates = torch.linspace(0, drop_path_probability, enc_depth).tolist()
        self.encoder_blocks = nn.ModuleList(
            [
                PDFormerEncoderBlock(
                    embed_dim,
                    geo_heads,
                    semantic_heads,
                    temporal_heads,
                    pattern_window,
                    spatial_chunk_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path_probability=drop_rates[index],
                )
                for index in range(enc_depth)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [nn.Conv2d(embed_dim, skip_dim, kernel_size=1) for _ in range(enc_depth)]
        )
        self.end_conv1 = nn.Conv2d(input_steps, output_steps, kernel_size=1)
        self.end_conv2 = nn.Conv2d(skip_dim, 1, kernel_size=1)

    def _input_patterns(self, values):
        signal = values[..., 0]
        batch, time_steps, nodes = signal.shape
        patterns = []
        for pattern_index in range(self.pattern_window):
            shift = self.pattern_window - 1 - pattern_index
            shifted = torch.zeros_like(signal)
            if shift == 0:
                shifted = signal
            else:
                shifted[:, shift:] = signal[:, : time_steps - shift]
            patterns.append(shifted)
        return torch.stack(patterns, dim=-1)

    def _run_block(self, block, x, x_patterns):
        args = (
            x,
            x_patterns,
            self.pattern_keys,
            self.geo_neighbors,
            self.geo_valid,
            self.semantic_neighbors,
        )
        if self.training and self.use_activation_checkpoint:
            if self.checkpoint_has_use_reentrant:
                return checkpoint(block, *args, use_reentrant=False)
            return checkpoint(block, *args)
        return block(*args)

    def forward(self, values, day_of_week):
        if values.shape[1] != self.input_steps or values.shape[2] != self.num_nodes:
            raise ValueError(
                f"Expected values (B,{self.input_steps},{self.num_nodes},1), got {tuple(values.shape)}"
            )
        x_patterns = self._input_patterns(values)
        encoded = self.embedding(values, day_of_week, self.lap_pe)
        skip = None
        for block, skip_conv in zip(self.encoder_blocks, self.skip_convs):
            encoded = self._run_block(block, encoded, x_patterns)
            current_skip = skip_conv(encoded.permute(0, 3, 2, 1))
            skip = current_skip if skip is None else skip + current_skip

        # Map the accumulated representation to the direct forecast horizon.
        output = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        output = self.end_conv2(F.relu(output.permute(0, 3, 2, 1)))
        return output.permute(0, 3, 2, 1).contiguous()


def build_pdformer(num_nodes, static_features):
    return ScalablePDFormer(
        num_nodes=num_nodes,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        geo_neighbors=static_features["geo_neighbors"],
        geo_valid=static_features["geo_valid"],
        semantic_neighbors=static_features["semantic_neighbors"],
        lap_pe=static_features["lap_pe"],
        pattern_keys=static_features["pattern_keys"],
        embed_dim=EMBED_DIM,
        skip_dim=SKIP_DIM,
        enc_depth=ENC_DEPTH,
        geo_heads=GEO_NUM_HEADS,
        semantic_heads=SEM_NUM_HEADS,
        temporal_heads=TEMP_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        drop=DROP,
        attn_drop=ATTN_DROP,
        drop_path_probability=DROP_PATH,
        pattern_window=S_ATTN_SIZE,
        spatial_chunk_size=SPATIAL_CHUNK_SIZE,
        use_activation_checkpoint=USE_ACTIVATION_CHECKPOINT,
    ).to(DEVICE)


pearson_corr_np = legacy_pearson


metric_all = legacy_metric_all


metric_nz = legacy_metric_nz


def empty_metric_store(num_layers, prediction_steps):
    keys = ["MAE_ALL", "COR_ALL", "RMSE_ALL", "MAE_NZ", "COR_NZ", "RMSE_NZ", "Zero_FP"]
    return {
        "layer": {
            layer: {day: {key: [] for key in keys} for day in range(prediction_steps)}
            for layer in range(num_layers)
        },
        "overall": {day: {key: [] for key in keys} for day in range(prediction_steps)},
    }


def append_metrics(bucket, true_np, pred_np):
    mae_all, cor_all, rmse_all, zero_fp = metric_all(true_np, pred_np)
    mae_nz, cor_nz, rmse_nz = metric_nz(true_np, pred_np)
    bucket["MAE_ALL"].append(mae_all)
    bucket["COR_ALL"].append(cor_all)
    bucket["RMSE_ALL"].append(rmse_all)
    bucket["MAE_NZ"].append(mae_nz)
    bucket["COR_NZ"].append(cor_nz)
    bucket["RMSE_NZ"].append(rmse_nz)
    bucket["Zero_FP"].append(zero_fp)


def save_results_csv(path, layers, prediction_steps, metric_store):
    metric_cols = ["MAE_ALL", "COR_ALL", "RMSE_ALL", "MAE_NZ", "COR_NZ", "RMSE_NZ", "Zero_FP"]
    rows = []
    for layer_index, layer in enumerate(layers):
        layer_days = {key: [] for key in metric_cols}
        for day in range(prediction_steps):
            values = {
                key: finite_mean(metric_store["layer"][layer_index][day][key])
                for key in metric_cols
            }
            rows.append({"Layer": layer, "Day": f"Day {day + 1}", **values})
            for key in metric_cols:
                layer_days[key].append(values[key])
        rows.append(
            {
                "Layer": layer,
                "Day": "AVERAGE",
                **{key: finite_mean(layer_days[key]) for key in metric_cols},
            }
        )

    overall_days = {key: [] for key in metric_cols}
    for day in range(prediction_steps):
        values = {key: finite_mean(metric_store["overall"][day][key]) for key in metric_cols}
        rows.append({"Layer": "OVERALL", "Day": f"Day {day + 1}", **values})
        for key in metric_cols:
            overall_days[key].append(values[key])
    rows.append(
        {
            "Layer": "OVERALL",
            "Day": "AVERAGE",
            **{key: finite_mean(overall_days[key]) for key in metric_cols},
        }
    )

    result = pd.DataFrame(rows)
    if len(result) != 40:
        raise RuntimeError(f"Unexpected result row count: {len(result)}")
    result.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    print(f"Saved result CSV: {path}")


def result_is_complete(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        result = pd.read_csv(path)
        required = {
            "Layer",
            "Day",
            "MAE_ALL",
            "COR_ALL",
            "RMSE_ALL",
            "MAE_NZ",
            "COR_NZ",
            "RMSE_NZ",
            "Zero_FP",
        }
        metrics = result[
            [
                "MAE_ALL",
                "COR_ALL",
                "RMSE_ALL",
                "MAE_NZ",
                "COR_NZ",
                "RMSE_NZ",
                "Zero_FP",
            ]
        ].to_numpy(dtype=float)
        return (
            len(result) == 40 and required.issubset(result.columns) and np.isfinite(metrics).all()
        )
    except Exception:
        return False


def inverse_normalize(layer_tensor, params):
    return layer_tensor * params["std"].to(layer_tensor.device) + params["mean"].to(
        layer_tensor.device
    )


def evaluate_pdformer(model, test_loader, layers, shapes, normalization_params):
    metric_store = empty_metric_store(len(layers), OUTPUT_STEPS)
    model.eval()
    use_amp = USE_AMP and DEVICE.type == "cuda"
    with torch.no_grad():
        for inputs, targets, day_of_week in test_loader:
            model_input = layers_to_node_tensor(inputs, DEVICE)
            day_of_week = day_of_week.to(DEVICE, non_blocking=True)
            with autocast_context(use_amp):
                prediction = model(model_input, day_of_week)
            prediction_layers = split_node_tensor(prediction.float(), shapes)

            for day in range(OUTPUT_STEPS):
                true_for_overall, pred_for_overall = [], []
                for layer_index in range(len(layers)):
                    target_norm = targets[layer_index][:, day, :, :, 0].to(DEVICE)
                    pred_norm = prediction_layers[layer_index][:, day, :, :, 0]
                    target_real = (
                        inverse_normalize(target_norm, normalization_params[layer_index])
                        .cpu()
                        .numpy()
                    )
                    pred_real = (
                        inverse_normalize(pred_norm, normalization_params[layer_index])
                        .cpu()
                        .numpy()
                    )

                    for batch_index in range(target_real.shape[0]):
                        append_metrics(
                            metric_store["layer"][layer_index][day],
                            target_real[batch_index],
                            pred_real[batch_index],
                        )
                    true_for_overall.append(target_real.reshape(target_real.shape[0], -1))
                    pred_for_overall.append(pred_real.reshape(pred_real.shape[0], -1))

                true_concat = np.concatenate(true_for_overall, axis=1)
                pred_concat = np.concatenate(pred_for_overall, axis=1)
                for batch_index in range(true_concat.shape[0]):
                    append_metrics(
                        metric_store["overall"][day],
                        true_concat[batch_index],
                        pred_concat[batch_index],
                    )
    return metric_store


def plot_loss(train_history, val_history, path, title):
    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(range(1, len(train_history) + 1), train_history, label="Train")
    plt.plot(range(1, len(val_history) + 1), val_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MAE")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def epoch_learning_rate(epoch):
    if epoch < WARMUP_EPOCHS:
        ratio = (epoch + 1) / max(WARMUP_EPOCHS, 1)
        return LEARNING_RATE * ratio
    progress = (epoch - WARMUP_EPOCHS) / max(NUM_EPOCHS - WARMUP_EPOCHS - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine


def load_state_dict_file(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def verify_layer_isolation(static_features):
    layer_ids = np.asarray(static_features["layer_ids"], dtype=np.int64)
    geo = np.asarray(static_features["geo_neighbors"], dtype=np.int64)
    geo_valid = np.asarray(static_features["geo_valid"], dtype=np.bool_)
    semantic = np.asarray(static_features["semantic_neighbors"], dtype=np.int64)
    source_layers = layer_ids[:, None]
    if np.any(layer_ids[geo][geo_valid] != np.broadcast_to(source_layers, geo.shape)[geo_valid]):
        raise RuntimeError("Geographic neighbors contain cross-layer links")
    if np.any(layer_ids[semantic] != source_layers):
        raise RuntimeError("Semantic neighbors contain cross-layer links")
    print("Layer-isolation check passed: geographic and semantic cross-layer links = 0")


def print_memory_estimate(num_nodes):
    bytes_per = 2 if DEVICE.type == "cuda" and USE_AMP else 4
    dense_elements = (
        BATCH_SIZE * INPUT_STEPS * (GEO_NUM_HEADS + SEM_NUM_HEADS) * num_nodes * num_nodes
    )
    sparse_elements = (
        BATCH_SIZE
        * INPUT_STEPS
        * num_nodes
        * (GEO_NUM_HEADS * ((2 * (GEO_HOP_THRESHOLD - 1) + 1) ** 2) + SEM_NUM_HEADS * SEMANTIC_K)
    )
    print(f"[Attention estimate] N={num_nodes}, dtype_bytes={bytes_per}")
    print(f"  Official dense spatial scores/block: {dense_elements * bytes_per / 1024**3:.2f} GiB")
    print(f"  Sparse allowed scores/block: {sparse_elements * bytes_per / 1024**3:.3f} GiB")
    print(f"  Runtime node chunk size: {SPATIAL_CHUNK_SIZE}")


def run_pdformer_city(seed, city):
    # 必须位于每个城市运行的最前面。
    set_seed(seed)

    run_name = f"{MODEL_NAME}_{city}"
    result_path = RESULT_ROOT / f"{city}_{MODEL_NAME}_Seed{seed}.csv"
    model_path = MODEL_ROOT / f"best_model_{run_name}_Seed{seed}.pth"
    if globals().get("EVALUATION_CHECKPOINT") is not None:
        model_path = Path(globals()["EVALUATION_CHECKPOINT"])
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    plot_path = PLOT_ROOT / f"{run_name}_loss_Seed{seed}.png"

    if SKIP_COMPLETED and result_is_complete(result_path) and model_path.exists():
        print(f"[SKIP] Completed result and best model exist: {city}, Seed{seed}")
        return

    layers, raw_levels = load_city_layers(city)
    total_timesteps = int(raw_levels[0].shape[0])
    train_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    shapes = [(int(data.shape[1]), int(data.shape[2])) for data in raw_levels]
    normalized_levels, normalization_params = normalize_data_levels(raw_levels, train_end)
    train_loader, val_loader, test_loader, train_end_check, _ = make_dataloaders(
        normalized_levels, total_timesteps
    )
    if train_end_check != train_end:
        raise RuntimeError("Training boundary mismatch")

    static_features = prepare_city_static_features(city, normalized_levels, train_end, shapes)
    verify_layer_isolation(static_features)
    num_nodes = sum(height * width for height, width in shapes)
    print_memory_estimate(num_nodes)

    print(f"\n{'=' * 100}")
    print(f"{MODEL_NAME} | city={city} | seed={seed}")
    print(f"Layers={layers}, shapes={shapes}, nodes={num_nodes}, timesteps={total_timesteps}")
    print(f"Train/Val/Test batches={len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print(f"Best model path: {model_path}")
    print(f"Result path: {result_path}")
    print(f"{'=' * 100}")

    model = build_pdformer(num_nodes, static_features)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    use_amp = USE_AMP and DEVICE.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_history, val_history = [], []

    for epoch in range(0 if globals().get("EVALUATION_CHECKPOINT") else NUM_EPOCHS):
        current_lr = epoch_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_batches = 0
        for batch_index, (inputs, targets, day_of_week) in enumerate(train_loader):
            model_input = layers_to_node_tensor(inputs, DEVICE)
            model_target = layers_to_node_tensor(targets, DEVICE)
            day_of_week = day_of_week.to(DEVICE, non_blocking=True)
            with autocast_context(use_amp):
                prediction = model(model_input, day_of_week)
                raw_loss = criterion(prediction, model_target)
                loss = raw_loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            train_loss_sum += float(raw_loss.detach())
            train_batches += 1

            should_step = (batch_index + 1) % GRAD_ACCUM_STEPS == 0 or batch_index + 1 == len(
                train_loader
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        average_train = train_loss_sum / max(train_batches, 1)
        train_history.append(average_train)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets, day_of_week in val_loader:
                model_input = layers_to_node_tensor(inputs, DEVICE)
                model_target = layers_to_node_tensor(targets, DEVICE)
                day_of_week = day_of_week.to(DEVICE, non_blocking=True)
                with autocast_context(use_amp):
                    prediction = model(model_input, day_of_week)
                    val_loss = criterion(prediction, model_target)
                val_loss_sum += float(val_loss)
                val_batches += 1
        average_val = val_loss_sum / max(val_batches, 1)
        val_history.append(average_val)
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"train={average_train:.6f} val={average_val:.6f} lr={current_lr:.2e}"
        )

        if average_val < best_val_loss - MIN_DELTA:
            best_val_loss = average_val
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if not model_path.exists():
        raise RuntimeError(f"Best validation checkpoint was not saved: {model_path}")
    if not globals().get("EVALUATION_CHECKPOINT"):
        plot_loss(train_history, val_history, plot_path, f"{run_name} Seed{seed}")

    # 最终测试必须严格加载验证集最佳权重。
    best_state = load_state_dict_file(model_path, DEVICE)
    model.load_state_dict(best_state, strict=True)
    print(f"Loaded best validation checkpoint: {model_path}")
    metric_store = evaluate_pdformer(model, test_loader, layers, shapes, normalization_params)
    save_results_csv(result_path, layers, OUTPUT_STEPS, metric_store)

    del model, optimizer, scaler, train_loader, val_loader, test_loader
    del raw_levels, normalized_levels, static_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_synthetic_sanity_check():
    set_seed(123)
    shapes = [(2, 2), (3, 3)]
    nodes = sum(height * width for height, width in shapes)
    geo_neighbors, geo_valid = build_geo_neighbors(shapes, hop_threshold=2)
    layer_ids = build_layer_ids(shapes)
    semantic = np.empty((nodes, min(3, nodes)), dtype=np.int64)
    for node in range(nodes):
        same_layer = np.flatnonzero(layer_ids == layer_ids[node])
        selected = same_layer[: min(3, len(same_layer))]
        if len(selected) < semantic.shape[1]:
            selected = np.pad(
                selected, (0, semantic.shape[1] - len(selected)), constant_values=node
            )
        semantic[node] = selected
    lap_pe = np.random.default_rng(1).normal(size=(nodes, 4)).astype(np.float32)
    pattern_keys = np.random.default_rng(2).normal(size=(6, S_ATTN_SIZE, 1)).astype(np.float32)
    model = ScalablePDFormer(
        num_nodes=nodes,
        input_steps=INPUT_STEPS,
        output_steps=OUTPUT_STEPS,
        geo_neighbors=geo_neighbors,
        geo_valid=geo_valid,
        semantic_neighbors=semantic,
        lap_pe=lap_pe,
        pattern_keys=pattern_keys,
        embed_dim=32,
        skip_dim=16,
        enc_depth=2,
        geo_heads=1,
        semantic_heads=1,
        temporal_heads=2,
        spatial_chunk_size=4,
        use_activation_checkpoint=False,
    ).to(DEVICE)
    values = torch.randn(2, INPUT_STEPS, nodes, 1, device=DEVICE)
    day_of_week = torch.arange(INPUT_STEPS, device=DEVICE).view(1, -1).expand(2, -1) % 7
    output = model(values, day_of_week)
    expected = (2, OUTPUT_STEPS, nodes, 1)
    if tuple(output.shape) != expected:
        raise RuntimeError(f"Sanity output {tuple(output.shape)} != {expected}")
    output.abs().mean().backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Sanity backward did not produce gradients")
    del model, values, output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Synthetic sanity check passed: output={expected}, forward/backward OK")
