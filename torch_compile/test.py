import torch
import transformer_engine.pytorch.cpp_extensions as tex

FP8_TENSOR_INDEX = 0

def bias_add_ln_fp8_native(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    w_shape: tuple,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: torch.float32,
    fp8_scale: torch.float32,
    output_dtype: torch.dtype,
    zero_centered_gamma: bool,
    scale_tensor: torch.Tensor,
    amax_tensor: torch.Tensor,
) -> torch.Tensor:
    # Bias-Add
    out1 = (x + bias) + residual
    out1 = out1.view(-1, w_shape[-1])

    # LayerNorm
    w = ln_weight if not zero_centered_gamma else 1 + ln_weight
    out2 = torch.nn.functional.layer_norm(out1, w_shape, w, ln_bias, eps)

    # Obtain FP8 amax
    amax = torch.amax(torch.abs(out2))
    amax_tensor.fill_(amax)

    # Apply FP8 scale and cast to FP8
    out2 = (out2 * scale_tensor).to(output_dtype)

    return out1, out2, amax_tensor

@torch.compile
def bias_add_ln_fp8_compile(*args) -> torch.Tensor:
    return bias_add_ln_fp8_native(*args)

def main():
    input_dtype = torch.bfloat16
    output_dtype = torch.float8_e4m3fn
    eps = 1e-5
    fp8_scale = 0.5
    x_shape = (512, 1, 12288)
    w_shape = (x_shape[-1],)
    zero_centered_gamma = True
    main_iters = 100

    # LayerNorm params
    ln_weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    ln_bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)

    # Inputs
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    bias = torch.randn(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    residual = torch.randn_like(x)

    '''
    meta = tex.FP8TensorMeta()
    meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
    meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / fp8_scale # TODO: Where is this used?
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
    '''

    for i in range(main_iters):
        '''
        # LayerNorm params
        ln_weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
        ln_bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)

        # Inputs
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)
        bias = torch.randn(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
        residual = torch.randn_like(x)
        '''

        #ln_weight_clone = ln_weight.detach().clone().requires_grad_()
        #ln_bias_clone = ln_bias.detach().clone().requires_grad_()
        x_clone = x.detach().clone().requires_grad_()
        bias_clone = bias.detach().clone().requires_grad_()
        residual_clone = residual.detach().clone().requires_grad_()

        meta = tex.FP8TensorMeta()
        meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
        meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / fp8_scale # TODO: Where is this used?
        meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")

        result = bias_add_ln_fp8_compile(
            x_clone,
            bias_clone,
            residual_clone,
            w_shape,
            ln_weight,
            ln_bias,
            eps,
            fp8_scale,
            output_dtype,
            zero_centered_gamma,
            meta.scale[FP8_TENSOR_INDEX],
            meta.amax_history[0][FP8_TENSOR_INDEX]
        )
        y = result[1].to(torch.bfloat16).mean()
        z = y.backward()

if __name__ == '__main__':
    main()
