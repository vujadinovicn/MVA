import os
import math
import torch
import pandas as pd

from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources

from softmax_matmul.softmax_matmul_sol import fused_softmax, softmax_mult

def log(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def run_benchmark_config(fn, nb_warmup, nb_passes):
    torch.cuda.reset_peak_memory_stats() # Reset peak memory before measurement

    # Warmup runs
    for _ in range(nb_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    # Run and measure function for nb_passes
    for _ in range(nb_passes):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)

    # Compute measures
    forward_ms_mean = sum(times) / len(times)
    forward_ms_std = math.sqrt(sum((t - forward_ms_mean) ** 2 for t in times) / len(times))
    forward_peak_MiB = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return forward_ms_mean, forward_ms_std, forward_peak_MiB

if __name__ == "__main__":
    # Create folders for logs and results
    os.makedirs("outputs/logs/", exist_ok=True) 
    os.makedirs("outputs/csv/", exist_ok=True)
    log_file = open("outputs/logs/softmax_matmul_benchmark.log", "w")

    # Parameters
    device = "cuda"
    dtype = torch.float32
    batch_size = 16
    nb_warmup = 10
    nb_passes = 100
    d1 = 2048
    d2_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    d3 = 512
    block_sizes = [16, 32, 64]

    results = []

    for d2_val in d2_values:
        # Initialize input tensors
        x = torch.randn(batch_size, d1, d2_val, device=device, dtype=dtype)
        V = torch.randn(batch_size, d2_val, d3, device=device, dtype=dtype)

        # Base csv row
        row = {"batch_size": batch_size, 
               "d1": d1, "d2": d2_val, "d3": d3, 
               "triton": False, "BLOCK": None, 
               "forward_ms_mean": None, "forward_ms_std": None, "forward_peak_MiB": None}
        
        # PyTorch implementation
        try:
            fn = lambda: softmax_mult(x, V)
            forward_ms_mean, forward_ms_std, forward_peak_MiB = run_benchmark_config(fn, nb_warmup=nb_warmup, nb_passes=nb_passes)
            row.update({
                "forward_ms_mean": forward_ms_mean, 
                "forward_ms_std": forward_ms_std,
                "forward_peak_MiB": forward_peak_MiB,
            })
            results.append(row)
            # log(f"[PyTorch] d2={d2_val} results: forward_ms_mean={forward_ms_mean:.3f}ms, forward_ms_std={forward_ms_std:.3f}ms, forward_peak_MiB={forward_peak_MiB:.1f}MiB")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"[PyTorch] OOM for d2={d2_val}")
                torch.cuda.empty_cache()
                results.append(row)
            else:
                raise

        # Triton fused kernel
        for BLOCK in block_sizes:
            # Base csv row
            row = {"batch_size": batch_size,
                   "d1": d1, "d2": d2_val, "d3": d3,
                   "triton": True, "BLOCK": BLOCK,
                   "forward_ms_mean": None, "forward_ms_std": None, "forward_peak_MiB": None}
            try:
                fn = lambda B=BLOCK: fused_softmax(x, V, BLOCK_1=B, BLOCK_2=B)
                forward_ms_mean, forward_ms_std, forward_peak_MiB = run_benchmark_config(fn, nb_warmup=nb_warmup, nb_passes=nb_passes)
                row.update({
                    "forward_ms_mean": forward_ms_mean, 
                    "forward_ms_std": forward_ms_std,
                    "forward_peak_MiB": forward_peak_MiB,
                })
                results.append(row)
                # log(f"[Triton] d2={d2_val}, B={BLOCK} results: forward_ms_mean={forward_ms_mean:.3f}ms, forward_ms_std={forward_ms_std:.3f}ms, forward_peak_MiB={forward_peak_MiB:.1f}MiB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log(f"[Triton] OOM for d2={d2_val}, BLOCK={BLOCK}")
                    torch.cuda.empty_cache()
                    results.append(row)
                else:
                    raise

            except OutOfResources:
                log(f"[Triton] OutOfResources for d2={d2_val}, BLOCK={BLOCK}")
                results.append(row)
                
            except CompileTimeAssertionFailure:
                log(f"=[Triton] Skipping BLOCK={BLOCK} inncompatible with d2={d2_val}")
                results.append(row)

    # Save results
    df = pd.DataFrame(results, columns=[
        "batch_size", "d1", "d2", "d3", "triton", "BLOCK",
        "forward_ms_mean", "forward_ms_std", "forward_peak_MiB",
    ])
    df["BLOCK"] = df["BLOCK"].astype("Int64")
    print(df.to_string())
    df.to_csv("outputs/csv/softmax_matmul_benchmark.csv", index=False)
    log_file.close()
