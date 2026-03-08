import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype)
        return x

if __name__ == "__main__":
    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
    rms_norm = RMSNorm().cuda()
    output = rms_norm(x)
    print(output.shape)