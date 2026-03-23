"""
RMSNorm with torch.compile — two modes for comparison.

Usage:
    # Default heuristic (fast compile, no autotune):
    python rms_norm.py

    # Max autotune (slower first call, better runtime):
    TORCH_TRACE="./my_inductor_log" python rms_norm.py --max-autotune

    # Dump inductor output code:
    TORCH_LOGS=output_code python rms_norm.py --max-autotune
"""

import argparse

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-autotune", action="store_true",
                        help="Enable max_autotune + coordinate descent tuning")
    args = parser.parse_args()

    model = RMSNorm().cuda()

    if args.max_autotune:
        model = torch.compile(
            model,
            dynamic=True,
            options={
                "max_autotune": True,
                "coordinate_descent_tuning": True,
                "triton.cudagraphs": True,
                "trace.enabled": True,
                "trace.graph_diagram": True,
            },
        )
    else:
        model = torch.compile(model, dynamic=True)

    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
    output = model(x)
    print(output.shape)