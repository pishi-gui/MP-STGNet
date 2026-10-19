"""ASTGCN baseline adapted for multilevel OSM tile-access forecasting."""

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


import re


import csv


import gc


import random


from collections import OrderedDict


from types import SimpleNamespace


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import Dataset, DataLoader


import torch.nn.functional as F


import matplotlib


import matplotlib.pyplot as plt


from scipy.sparse.linalg import eigs


DATA_ROOT = REPO_ROOT / "data"

RESULT_ROOT = REPO_ROOT / "runs/baselines/results"

MODEL_ROOT = REPO_ROOT / "runs/baselines/checkpoints"

PLOT_ROOT = REPO_ROOT / "runs/baselines/figures"

RUN_CITIES = ["Rome", "Milan", "Vienna", "SanFrancisco", "Seoul"]

TRAIN_SPLIT_RATIO = 0.6


VALIDATION_SPLIT_RATIO = 0.2


TEST_SPLIT_RATIO = 0.2


INPUT_STEPS = 28


OUTPUT_STEPS = 7


BATCH_SIZE = 4


NUM_EPOCHS = 200


PATIENCE = 10


MIN_DELTA = 1e-4


BASE_SEED = 76


SKIP_COMPLETED = True


DEVICE = torch.device("cpu")  # Resolved explicitly by run_baseline.py.


def set_seed(seed=99):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed at {seed}")


def load_city_layers(city, num_layers=4):
    if not 1 <= num_layers <= 4:
        raise ValueError("num_layers must be between 1 and 4")
    arrays, _ = load_city(DATA_ROOT, city)
    return [14, 15, 16, 17][:num_layers], arrays[:num_layers]


def normalize_data_levels(data_list, train_time_end, epsilon=1e-5):
    normalized = []
    params = []
    for data in data_list:
        train_slice = data[:train_time_end]
        mean = train_slice.mean()
        std = train_slice.std()
        if std < epsilon:
            std = torch.tensor(epsilon, dtype=data.dtype)
        normalized.append((data - mean) / std)
        params.append({"mean": mean, "std": std})
    return normalized, params


def inverse_normalize_data_levels(normalized_data_list, normalization_params_list):
    out = []
    for data, params in zip(normalized_data_list, normalization_params_list):
        out.append(data * params["std"].to(data.device) + params["mean"].to(data.device))
    return out


class MultiLayerSpatioTemporalDataset(Dataset):
    def __init__(self, data_list, input_steps, output_steps, time_start=0, time_end=None):
        self.data_list = data_list
        self.input_steps = input_steps
        self.output_steps = output_steps
        T = data_list[0].shape[0]
        if time_end is None:
            time_end = T
        max_start = time_end - (input_steps + output_steps)
        if max_start < time_start:
            raise ValueError(
                f"Time range [{time_start}, {time_end}) is too short for "
                f"input_steps={input_steps}, output_steps={output_steps}."
            )
        self.valid_indices = list(range(time_start, max_start + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.input_steps
        target_end = end + self.output_steps
        inputs = []
        targets = []
        for data in self.data_list:
            inputs.append(data[start:end])
            targets.append(data[end:target_end])
        return inputs, targets


def make_dataloaders(normalized_data_levels, total_timesteps):
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    val_time_end = int(total_timesteps * (TRAIN_SPLIT_RATIO + VALIDATION_SPLIT_RATIO))

    train_dataset = MultiLayerSpatioTemporalDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, 0, train_time_end
    )
    val_dataset = MultiLayerSpatioTemporalDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, train_time_end, val_time_end
    )
    test_dataset = MultiLayerSpatioTemporalDataset(
        normalized_data_levels, INPUT_STEPS, OUTPUT_STEPS, val_time_end, total_timesteps
    )
    kwargs = dict(
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    return (
        DataLoader(train_dataset, **kwargs),
        DataLoader(val_dataset, **kwargs),
        DataLoader(test_dataset, **kwargs),
        train_time_end,
        val_time_end,
    )


def layers_list_to_model_input(layers, device=None):
    tensors = []
    B = T = C = None
    for idx, layer in enumerate(layers):
        x = layer if isinstance(layer, torch.Tensor) else torch.as_tensor(layer)
        if device is not None:
            x = x.to(device)
        if x.dim() != 5:
            raise ValueError(f"Layer {idx} must be (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        if idx == 0:
            B, T, C = b, t, c
        elif (b, t, c) != (B, T, C):
            raise ValueError("Layer batch/time/channel sizes are inconsistent.")
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, h * w, C, T)
        tensors.append(x)
    return torch.cat(tensors, dim=1)


def model_input_to_layers_list(x, shapes):
    if x.dim() != 3:
        raise ValueError(f"x must be (B,N,C), got {tuple(x.shape)}")
    B, N, C = x.shape
    expected = sum(h * w for h, w in shapes)
    if N != expected:
        raise ValueError(f"N={N} does not match shapes total={expected}")
    out = []
    ptr = 0
    for h, w in shapes:
        n = h * w
        out.append(x[:, ptr : ptr + n, :].view(B, h, w, C).contiguous())
        ptr += n
    return out


def shapes_to_adjacency(shapes, self_loop=True, normalize=True):
    idx_maps = []
    offsets = []
    idx = 0
    for h, w in shapes:
        offsets.append(idx)
        idx_maps.append((np.arange(h * w).reshape(h, w) + idx).astype(np.int32))
        idx += h * w
    A = np.zeros((idx, idx), dtype=np.float32)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for layer_idx, (h, w) in enumerate(shapes):
        imap = idx_maps[layer_idx]
        for r in range(h):
            for c in range(w):
                u = int(imap[r, c])
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        A[u, int(imap[rr, cc])] = 1.0
    if self_loop:
        np.fill_diagonal(A, 1.0)
    if normalize:
        deg = A.sum(axis=1)
        deg[deg == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        A = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    return A


metric_triplet = legacy_metric_triplet


def result_is_complete(path):
    if not Path(path).exists():
        return False
    try:
        df = pd.read_csv(path)
        required = {"Layer", "Day", "MAE", "MAPE_floor1", "RMSE"}
        return (
            len(df) == 40
            and required.issubset(df.columns)
            and np.isfinite(df[["MAE", "MAPE_floor1", "RMSE"]].to_numpy(dtype=float)).all()
        )
    except Exception:
        return False


def save_results_csv(path, layers, prediction_steps, layer_metrics, overall_metrics):
    rows = []
    for i, layer in enumerate(layers):
        layer_mae, layer_mape, layer_rmse = [], [], []
        for d in range(prediction_steps):
            mae = float(np.mean(layer_metrics[i][d]["MAE"]))
            mape = float(np.mean(layer_metrics[i][d]["MAPE_floor1"]))
            rmse = float(np.mean(layer_metrics[i][d]["RMSE"]))
            rows.append(
                {
                    "Layer": layer,
                    "Day": f"Day {d + 1}",
                    "MAE": mae,
                    "MAPE_floor1": mape,
                    "RMSE": rmse,
                }
            )
            layer_mae.append(mae)
            layer_mape.append(mape)
            layer_rmse.append(rmse)
        rows.append(
            {
                "Layer": layer,
                "Day": "AVERAGE",
                "MAE": float(np.mean(layer_mae)),
                "MAPE_floor1": float(np.mean(layer_mape)),
                "RMSE": float(np.mean(layer_rmse)),
            }
        )
    overall_mae, overall_mape, overall_rmse = [], [], []
    for d in range(prediction_steps):
        mae = float(np.mean(overall_metrics[d]["MAE"]))
        mape = float(np.mean(overall_metrics[d]["MAPE_floor1"]))
        rmse = float(np.mean(overall_metrics[d]["RMSE"]))
        rows.append(
            {
                "Layer": "OVERALL",
                "Day": f"Day {d + 1}",
                "MAE": mae,
                "MAPE_floor1": mape,
                "RMSE": rmse,
            }
        )
        overall_mae.append(mae)
        overall_mape.append(mape)
        overall_rmse.append(rmse)
    rows.append(
        {
            "Layer": "OVERALL",
            "Day": "AVERAGE",
            "MAE": float(np.mean(overall_mae)),
            "MAPE_floor1": float(np.mean(overall_mape)),
            "RMSE": float(np.mean(overall_rmse)),
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    print(f"Saved result CSV: {path}")


def plot_loss(train_loss_history, val_loss_history, path, title):
    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def scaled_Laplacian(W):
    """
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    """

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which="LR")[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    """

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        """

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
                for _ in range(K)
            ]
        )

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE
            )  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(
                    spatial_attention
                )  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(
                    0, 2, 1
                ).matmul(
                    graph_signal
                )  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
                for _ in range(K)
            ]
        )

    def forward(self, x):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE
            )  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    def __init__(
        self,
        DEVICE,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_of_vertices,
        num_of_timesteps,
    ):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(
            batch_size, num_of_vertices, num_of_features, num_of_timesteps
        )

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3)
        )  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(
            x.permute(0, 2, 1, 3)
        )  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(
            0, 2, 3, 1
        )
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCN_submodule(nn.Module):
    def __init__(
        self,
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices,
    ):
        """
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        """

        super(ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList(
            [
                ASTGCN_block(
                    DEVICE,
                    in_channels,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    time_strides,
                    cheb_polynomials,
                    num_of_vertices,
                    len_input,
                )
            ]
        )

        self.BlockList.extend(
            [
                ASTGCN_block(
                    DEVICE,
                    nb_time_filter,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    cheb_polynomials,
                    num_of_vertices,
                    len_input // time_strides,
                )
                for _ in range(nb_block - 1)
            ]
        )

        self.final_conv = nn.Conv2d(
            int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter)
        )

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        """
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        """
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


def make_model(
    DEVICE,
    nb_block,
    in_channels,
    K,
    nb_chev_filter,
    nb_time_filter,
    time_strides,
    adj_mx,
    num_for_predict,
    len_input,
    num_of_vertices,
):
    """

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    """
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [
        torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)
    ]
    model = ASTGCN_submodule(
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model


def run_astgcn_city(city):
    set_seed(BASE_SEED)
    name = f"ASTGCN_Baseline_{city}"
    result_path = RESULT_ROOT / f"{city}_ASTGCN_Baseline_Seed{BASE_SEED}.csv"
    model_path = MODEL_ROOT / f"best_model_{name}_Seed{BASE_SEED}.pth"
    if globals().get("EVALUATION_CHECKPOINT") is not None:
        model_path = Path(globals()["EVALUATION_CHECKPOINT"])
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    plot_path = PLOT_ROOT / f"{name}_loss_Seed{BASE_SEED}.png"
    if SKIP_COMPLETED and result_is_complete(result_path):
        print(f"[SKIP] Completed result exists: {result_path}")
        return

    layers, data_list = load_city_layers(city)
    total_timesteps = data_list[0].shape[0]
    shapes = [(int(x.shape[1]), int(x.shape[2])) for x in data_list]
    train_time_end = int(total_timesteps * TRAIN_SPLIT_RATIO)
    normalized, params = normalize_data_levels(data_list, train_time_end)
    train_loader, val_loader, test_loader, _, _ = make_dataloaders(normalized, total_timesteps)

    print(f"\n=== ASTGCN baseline | {city} ===")
    print(f"Layers={layers}, shapes={shapes}")
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    nb_block = 2
    in_channels = 1
    K = 3
    nb_chev_filter = 64
    nb_time_filter = 64
    time_strides = 1
    num_for_predict = 1
    len_input = INPUT_STEPS
    adj_mx = shapes_to_adjacency(shapes)
    num_nodes = sum(h * w for h, w in shapes)
    model = make_model(
        DEVICE=DEVICE,
        nb_block=nb_block,
        in_channels=in_channels,
        K=K,
        nb_chev_filter=nb_chev_filter,
        nb_time_filter=nb_time_filter,
        time_strides=time_strides,
        adj_mx=adj_mx,
        num_for_predict=num_for_predict,
        len_input=len_input,
        num_of_vertices=num_nodes,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(0 if globals().get("EVALUATION_CHECKPOINT") else NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs_list_batch, targets_list_batch in train_loader:
            inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
            # One-step training; Day1 target only. Day1-Day7 is evaluated recursively.
            targets_one = [y[:, 0:1].to(DEVICE) for y in targets_list_batch]
            inputs_model = layers_list_to_model_input(inputs_list_batch)
            targets_model = layers_list_to_model_input(targets_one).squeeze(-1)

            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs_model)
            loss = criterion(preds, targets_model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        train_loss_history.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_list_batch, targets_list_batch in val_loader:
                inputs_list_batch = [x.to(DEVICE) for x in inputs_list_batch]
                targets_one = [y[:, 0:1].to(DEVICE) for y in targets_list_batch]
                inputs_model = layers_list_to_model_input(inputs_list_batch)
                targets_model = layers_list_to_model_input(targets_one).squeeze(-1)
                preds = model(inputs_model)
                val_loss += criterion(preds, targets_model).item()
        avg_val = val_loss / len(val_loader)
        val_loss_history.append(avg_val)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] train={avg_train:.6f} val={avg_val:.6f}")

        if avg_val < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if not globals().get("EVALUATION_CHECKPOINT"):
        plot_loss(train_loss_history, val_loss_history, plot_path, f"{name} loss")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    prediction_steps = OUTPUT_STEPS
    layer_metrics = {
        i: {d: {"MAE": [], "MAPE_floor1": [], "RMSE": []} for d in range(prediction_steps)}
        for i in range(len(layers))
    }
    overall_metrics = {
        d: {"MAE": [], "MAPE_floor1": [], "RMSE": []} for d in range(prediction_steps)
    }

    with torch.no_grad():
        for inputs_list_batch, targets_list_batch in test_loader:
            current_inputs = [x.to(DEVICE) for x in inputs_list_batch]
            targets_list_batch = [y.to(DEVICE) for y in targets_list_batch]
            batch_preds_denorm = []
            for d in range(prediction_steps):
                inputs_model = layers_list_to_model_input(current_inputs)
                preds = model(inputs_model)
                pred_layers = model_input_to_layers_list(preds, shapes)
                step_denorm = []
                for i in range(len(layers)):
                    pred_norm = pred_layers[i]
                    pred_denorm = inverse_normalize_data_levels([pred_norm], [params[i]])[0]
                    step_denorm.append(pred_denorm)
                batch_preds_denorm.append(step_denorm)
                if d < prediction_steps - 1:
                    current_inputs = [
                        torch.cat([current_inputs[i][:, 1:], pred_layers[i].unsqueeze(1)], dim=1)
                        for i in range(len(layers))
                    ]

            for d in range(prediction_steps):
                true_flat_parts = []
                pred_flat_parts = []
                for i in range(len(layers)):
                    true_norm = targets_list_batch[i][:, d]
                    true_denorm = inverse_normalize_data_levels([true_norm], [params[i]])[0]
                    pred_denorm = batch_preds_denorm[d][i]
                    true_np = true_denorm.cpu().numpy()
                    pred_np = pred_denorm.cpu().numpy()
                    for k in range(true_np.shape[0]):
                        mae, mape, rmse = metric_triplet(true_np[k], pred_np[k])
                        layer_metrics[i][d]["MAE"].append(mae)
                        layer_metrics[i][d]["MAPE_floor1"].append(mape)
                        layer_metrics[i][d]["RMSE"].append(rmse)
                    true_flat_parts.append(true_np.reshape(true_np.shape[0], -1))
                    pred_flat_parts.append(pred_np.reshape(pred_np.shape[0], -1))
                true_concat = np.concatenate(true_flat_parts, axis=1)
                pred_concat = np.concatenate(pred_flat_parts, axis=1)
                for k in range(true_concat.shape[0]):
                    mae, mape, rmse = metric_triplet(true_concat[k], pred_concat[k])
                    overall_metrics[d]["MAE"].append(mae)
                    overall_metrics[d]["MAPE_floor1"].append(mape)
                    overall_metrics[d]["RMSE"].append(rmse)

    save_results_csv(result_path, layers, prediction_steps, layer_metrics, overall_metrics)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
