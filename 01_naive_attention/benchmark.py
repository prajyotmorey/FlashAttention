from typing import Sequence

import torch
import torch.nn as nn
from transformers.cli.serving import model_manager
from naive_attention import NaiveAttention
import math


def attention_memory(batch_size, N, hidden_dim, dtype):
    total_elements = batch_size * hidden_dim * N * N  # (B,H,N,N)---actual size of attention
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    attn_mem = bytes_per_element * total_elements
    return attn_mem / (1024 ** 2)


def gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    return allocated, reserved


## Below functions are for Timming alanysis.
def benchmark_attention(model, x, warmup=10, iterations=50):
    model.eval()

    with torch.no_grad():  # Warmup
        for _ in range(warmup):
            model(x)

    torch.cuda.synchronize()

    # Initialize: start-end timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    with torch.no_grad():
        for _ in range(iterations):
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    times.sort()
    median_time = times[len(times) // 2]

    return median_time


def detailed_timing(model, x, warmup=10, iterations=50):
    model.eval()
    B, N, _ = x.shape
    H = model.head_num
    D = model.head_dim

    with torch.no_grad():
        for _ in range(warmup):
            Q = model.Wq(x)
            K = model.Wk(x)
            V = model.Wv(x)

            Q = Q.view(B, N, H, D).transpose(1, 2)
            K = K.view(B, N, H, D).transpose(1, 2)
            V = V.view(B, N, H, D).transpose(1, 2)

            scores = torch.matmul(
                Q,
                K.transpose(-2, -1)
            )

            scores = scores / math.sqrt(D)

            attention = torch.softmax(scores, dim=-1)

            output = torch.matmul(attention, V)

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    qkv_times = []
    qk_times = []
    softmax_times = []
    pv_times = []

    with torch.no_grad():

        for _ in range(iterations):
            # QKV
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            Q = model.Wq(x)
            K = model.Wk(x)
            V = model.Wv(x)

            Q = Q.view(B, N, H, D).transpose(1, 2)
            K = K.view(B, N, H, D).transpose(1, 2)
            V = V.view(B, N, H, D).transpose(1, 2)

            end.record()
            torch.cuda.synchronize()

            qkv_times.append(start.elapsed_time(end))

            # ------------------------------------------------
            # QK^T
            # ------------------------------------------------

            start.record()

            scores = torch.matmul(
                Q,
                K.transpose(-2, -1)
            )

            scores = scores / math.sqrt(D)

            end.record()
            torch.cuda.synchronize()

            qk_times.append(start.elapsed_time(end))

            # ------------------------------------------------
            # Softmax
            # ------------------------------------------------

            start.record()

            attention = torch.softmax(
                scores,
                dim=-1
            )

            end.record()
            torch.cuda.synchronize()

            softmax_times.append(start.elapsed_time(end))

            # ------------------------------------------------
            # P @ V
            # ------------------------------------------------

            start.record()

            output = torch.matmul(
                attention,
                V
            )

            end.record()
            torch.cuda.synchronize()

            pv_times.append(start.elapsed_time(end))

    def median(values):

        values = sorted(values)

        return values[len(values) // 2]

    return {
        "QKV_ms": median(qkv_times),
        "QK_ms": median(qk_times),
        "Softmax_ms": median(softmax_times),
        "PV_ms": median(pv_times),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model inputs
    batch_size = 2
    hidden_dim = 512
    head_num = 8
    head_dim = hidden_dim // head_num

    # Analyse over the---
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    # sequence_lengths = [128, 256]

    dtype = torch.float16
    print("=" * 80)
    print("Naive Attention Benchmark")
    print("=" * 80)

    print(f"GPU       : {torch.cuda.get_device_name()}")
    print(f"B         : {batch_size}")
    print(f"H         : {head_num}")
    print(f"D         : {head_dim}")
    print(f"d_model   : {hidden_dim}")
    print(f"dtype     : {dtype}")
    print()

    print(
        f"{'N':>6} "
        f"{'Attn Mem(MB)':>15} "
        f"{'Peak GPU Mem(MB)':>20}"
        f"{'QKV(ms)':>12} "
        f"{'QK(ms)':>12} "
        f"{'Softmax(ms)':>14} "
        f"{'PV(ms)':>12} "
        f"{'Total(ms)':>14}"
        f"{"Model Time(ms)":>17}"
    )

    print("-" * 150)

    for N in sequence_lengths:
        # print(f"Running for sequence lenght(N): {N}")

        model = NaiveAttention(d_model=hidden_dim, head_num=head_num, head_dim=head_dim).to(device).to(dtype)
        x = torch.randn(batch_size, N, hidden_dim).to(device).to(dtype)

        # Analytical attnetion memory
        attention_mem = attention_memory(batch_size, N, hidden_dim, dtype)

        ##Clear memory statistics
        torch.cuda.empty_cache()  # empty cache
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory

        timing = detailed_timing(model, x, warmup=10, iterations=50)

        total_time = (timing["QKV_ms"] + timing["QK_ms"] + timing["Softmax_ms"] + timing["PV_ms"])

        # Model Benchmark--Whole Model
        median_time = benchmark_attention(model, x, warmup=10, iterations=50)

        allocated, reserve = gpu_memory()

        # Actual peak memory
        peak_memory = (torch.cuda.max_memory_allocated() / (1024 ** 2))

        print(
            f"{N:6d} "
            f"{attention_mem:15.2f} "
            f"{peak_memory:20.2f}"
            f"{timing['QKV_ms']:12.3f} "
            f"{timing['QK_ms']:12.3f} "
            f"{timing['Softmax_ms']:14.3f} "
            f"{timing['PV_ms']:12.3f} "
            f"{total_time:14.3f}"
            f"{median_time:17.2}"
        )

        # clean uop device.
        del model
        del x

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()