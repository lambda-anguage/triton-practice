"""
not finished
"""
import torch
import triton
import triton.language as tl

from benchmark import bench

@triton.jit
def quantize(x_ptr, fp8_ptr, scale_ptr, m, n, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_m = pid // m
    pid_n = pid % m
    offset = pid * block_size
    indices = offset + tl.arange(0, block_size)
    mask = indices < n

    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)
    z = x + y
    tl.store(z_ptr + indices, z, mask=mask)

def torch_fn(x):
    assert x.dim() == 2
    m, n = x.shape
    assert n % 128 == 0
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

def triton_fn(x, y):
    assert x.dim() == 2
    m, n = x.shape
    assert n % 128 == 0
    fp8 = torch.empty(m, n // 128, 128, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(m, n // 128, device=x.device, dtype=torch.float)
    _add[grid](x, y, z, n, block_size=128)
    return z

dtype = torch.float
device = "cuda"

x = torch.randn(4096, 2048, device=device, dtype=dtype)
y = torch.randn(4096, 2048, device=device, dtype=dtype)
assert torch.allclose(
    torch_fn(x, y),
    triton_fn(x, y),
)

fns = {
    "torch": torch_fn,
    "triton": triton_fn,
}

for k, v in fns.items():
    time = bench(v, x, y)
    print(f"{k} benchmark: {time * 1e6:.2f}ms")
