import torch

@torch.compile
def ln_fp8(
    x: torch.Tensor,
    w_shape: tuple,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: torch.float32,
    output_dtype: torch.dtype
) -> torch.Tensor:
    out = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(output_dtype)
    return out

input_dtype = torch.bfloat16
output_dtype = torch.float8_e4m3fn
eps = 1e-5

x_shape = (1024, 1024)
w_shape = (x_shape[-1],)
weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)

y = ln_fp8(x, w_shape, weight, bias, eps, output_dtype)
print(y)
