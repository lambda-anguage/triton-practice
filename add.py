import torch
import triton
import triton.language as tl

from benchmark import bench

@triton.jit
def _add(x_ptr, y_ptr, z_ptr, n, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * block_size
    indices = offset + tl.arange(0, block_size)
    mask = indices < n

    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)
    z = x + y
    tl.store(z_ptr + indices, z, mask=mask)

def torch_fn(x, y):
    return x + y

def triton_fn(x, y):
    z = torch.empty_like(x)
    n = z.numel()
    grid = lambda meta: (triton.cdiv(n, meta['block_size']),)
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
