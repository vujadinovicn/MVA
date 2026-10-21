import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd

from flash_attention.flash_attention import FlashAttentionTriton

def log(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def measure_saved_activations(forward_fn, Q, K, V):
    Q_copy = Q.detach().requires_grad_(True)
    K_copy = K.detach().requires_grad_(True)
    V_copy = V.detach().requires_grad_(True)
    output = forward_fn(Q_copy, K_copy, V_copy)

    mem = 0
    if output.grad_fn is not None:
        for t in output.grad_fn.saved_tensors:
            mem += t.nelement() * t.element_size()
    saved_activations_MiB = mem / (1024 ** 2)

    # Clean up
    del output, Q_copy, K_copy, V_copy
    torch.cuda.empty_cache()

    return saved_activations_MiB

def forward_backward_fn(forward_fn, Q, K, V):
    Q_g = Q.detach().requires_grad_(True)
    K_g = K.detach().requires_grad_(True)
    V_g = V.detach().requires_grad_(True)
    out = forward_fn(Q_g, K_g, V_g)
    out.sum().backward()

def run_benchmark_config(forward_fn, Q, K, V, nb_warmup, nb_passes, forward=True):
    torch.cuda.reset_peak_memory_stats()

    # Warmup: Run either forward or forward and backward
    for _ in range(nb_warmup):
        if forward: 
            forward_fn(Q, K, V)
        else:
            forward_backward_fn(forward_fn, Q, K, V)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    # Run and measure function for nb_passes
    for _ in range(nb_passes):
        if not forward:
            # If backward, run forward outside measurement
            Q_copy = Q.detach().requires_grad_(True)
            K_copy = K.detach().requires_grad_(True)
            V_copy = V.detach().requires_grad_(True)
            out = forward_fn(Q_copy, K_copy, V_copy)

        torch.cuda.synchronize()
        start.record()

        # Run either forward or backward
        if forward:
            forward_fn(Q, K, V)
        else:
            out.sum().backward()
        end.record()
        torch.cuda.synchronize()
                
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)

    ms_mean = sum(times) / len(times)
    peak_MiB = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ms_mean, peak_MiB

def benchmark_impl(impl, forward_fn, Q, K, V, nb_warmup, nb_forward_passes, nb_backward_passes):
    result = {
        "implementation": impl,
        "forward_ms": None,
        "forward_peak_MiB": None,
        "backward_ms": None,
        "backward_peak_MiB": None,
        "saved_activations_MiB": None,
        "status": "ok",
    }

    # Forward pass
    try:
        with torch.no_grad():
            forward_ms, forward_peak_MiB = run_benchmark_config(forward_fn, Q, K, V, nb_warmup, nb_forward_passes, forward=True)
        result["forward_ms"] = forward_ms
        result["forward_peak_MiB"] = forward_peak_MiB
    except torch.cuda.OutOfMemoryError:
        log(f"[{impl}] OOM forward")
        torch.cuda.empty_cache()
        result["status"] = "OOM"
        return result

    # Saved activations
    result["saved_activations_MiB"] = measure_saved_activations(forward_fn, Q, K, V)

    # Backward pass
    try:
        backward_ms, backward_peak_MiB = run_benchmark_config(forward_fn, Q, K, V, nb_warmup, nb_backward_passes, forward=False)
        result["backward_ms"] = backward_ms
        result["backward_peak_MiB"] = backward_peak_MiB
    except torch.cuda.OutOfMemoryError:
        log(f"[{impl}] OOM backward")
        torch.cuda.empty_cache()
        result["status"] = "OOM(backward)"

    return result

# Attention wrappers
def pytorch_sdpa(Q, K, V):
    Q_ = Q.unsqueeze(1)
    K_ = K.unsqueeze(1)
    V_ = V.unsqueeze(1)
    return F.scaled_dot_product_attention(Q_, K_, V_).squeeze(1)

def flash_triton(Q, K, V):
    return FlashAttentionTriton.apply(Q, K, V)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["pytorch_sdpa", "flash_triton"], default=None, help="Default: Run all implementations")
    args = parser.parse_args()

    # Create folders for logs and results
    os.makedirs("outputs/logs/", exist_ok=True)
    os.makedirs("outputs/csv/", exist_ok=True)
    log_file = open("outputs/logs/attention_benchmark.log", "w")

    # Parameters
    device = "cuda"
    dtype = torch.float32
    batch_size = 8
    nb_warmup = 10
    nb_forward_passes = 100
    nb_backward_passes = 100
    d_models = [64]
    context_lengths = [256, 1024, 4096, 8192, 16384]

    gpu_name = torch.cuda.get_device_name(0)
    log(f"GPU name: {gpu_name}")

    # Compiled implementations
    pytorch_sdpa_compiled = torch.compile(pytorch_sdpa)
    flash_triton_compiled = torch.compile(flash_triton)

    # Select implementations
    if args.impl is None:
        impls = {
            "pytorch_sdpa": pytorch_sdpa_compiled,
            "flash_triton": flash_triton_compiled,
        }
    elif args.impl == "pytorch_sdpa":
        impls = {"pytorch_sdpa": pytorch_sdpa_compiled}
    else:
        impls = {"flash_triton": flash_triton_compiled}
    log(f"Running benchmark for implementations: {', '.join(impls.keys())}")

    results = []

    for d_model in d_models:
        for seq_len in context_lengths:
            Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
            K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
            V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

            for impl, forward_fn in impls.items():
                row = benchmark_impl(impl, forward_fn, Q, K, V, nb_warmup, nb_forward_passes, nb_backward_passes)
                row["d_model"] = d_model
                row["seq_len"] = seq_len
                row["gpu"] = gpu_name

                results.append(row)

                if not row["status"] == "ok":
                    log(f"[{impl}] d_model={d_model}, seq_len={seq_len}: status={row['status']}")

            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results, columns=[
        "implementation", "d_model", "seq_len",
        "forward_ms", "forward_peak_MiB", "backward_ms", 
        "backward_peak_MiB", "saved_activations_MiB", 
        "status", "gpu",
    ])
    print(df.to_string())
    df.to_csv("outputs/csv/attention_benchmark.csv", index=False)
    log_file.close()