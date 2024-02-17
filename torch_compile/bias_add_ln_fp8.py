import torch

@torch.compile
def bias_add_ln_fp8(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    w_shape: tuple,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: torch.float32,
    fp8_scale: torch.float32,
    output_dtype: torch.dtype
) -> torch.Tensor:
    # Bias-Add
    out1 = (x + bias) + residual

    # LayerNorm
    out2 = torch.nn.functional.layer_norm(out1, w_shape, ln_weight, ln_bias, eps)

    # Obtain FP8 amax
    amax = torch.max(torch.abs(out2)).to(torch.float32)

    # Apply FP8 scale and cast to FP8
    out2 = (out2 * fp8_scale).to(output_dtype)

    return out1, out2, amax

# Settings - BF16 input, FP8 output
input_dtype = torch.bfloat16
output_dtype = torch.float8_e4m3fn
eps = 1e-5
fp8_scale = 0.5

# Shapes
x_shape = (1024, 1024)
w_shape = (x_shape[-1],)

# LayerNorm params
ln_weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
ln_bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)

# Inputs
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)
bias = torch.randn_like(x)
residual = torch.randn_like(x)

# Fused Bias-Add-LayerNorm-FP8
y1, y2, amax = bias_add_ln_fp8(x, bias, residual, w_shape, ln_weight, ln_bias, eps, fp8_scale, output_dtype)

print(y1)
print(y2)
print(amax)
