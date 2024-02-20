import torch
import transformer_engine.pytorch.cpp_extensions as tex

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
) -> torch.Tensor:
    # Bias-Add
    out1 = (x + bias) + residual

    # LayerNorm
    out2 = torch.nn.functional.layer_norm(out1.view(-1, w_shape[-1]), w_shape, ln_weight, ln_bias, eps)

    # Obtain FP8 amax
    amax = torch.max(torch.abs(out2)).to(torch.float32)

    # Apply FP8 scale and cast to FP8
    out2 = (out2 * fp8_scale).to(output_dtype)

    return out1, out2, amax

@torch.compile
def bias_add_ln_fp8_compile(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    w_shape: tuple,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: torch.float32,
    fp8_scale: torch.float32,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return bias_add_ln_fp8_native(x,
                                  bias,
                                  residual,
                                  w_shape,
                                  ln_weight,
                                  ln_bias,
                                  eps,
                                  fp8_scale,
                                  output_dtype)

@torch.compile
def bias_dropout_add_fused(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    """dropout(inp + bias) + residual"""
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out

def bias_add_ln_fp8_te(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    w_shape: tuple,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: torch.float32,
    fp8_scale: torch.float32,
    output_dtype: torch.dtype,
    meta: tex.FP8TensorMeta,
) -> torch.Tensor:
    bda_out = bias_dropout_add_fused(x, bias, residual, 0, True)
    bda_out = bda_out.view(-1, w_shape[-1])

    ln_in = (bda_out, ln_weight, ln_bias)
    ln_out = torch.empty_like(bda_out, dtype=torch.uint8)
    ln_out_kwarg = {"ln_out": ln_out}
    out = tex.layernorm_fwd_fp8(*ln_in,
                                eps,
                                meta,
                                0, # tex.FP8FwdTensors.GEMM1_INPUT
                                tex.DType.kFloat8E4M3, # fp8_dtype_forward
                                0, # TODO: fwd_ln_sm_margin
                                True, # zero_centered_gamma
                                **ln_out_kwarg,
    )
    return out

def bias_add_ln_fp8(*args, **kwargs) -> torch.Tensor:
    impl = kwargs["impl"]

    if impl == "native":
        result = bias_add_ln_fp8_native(*args)
    elif impl == "compile":
        result = bias_add_ln_fp8_compile(*args)
    elif impl == "te":
        result = bias_add_ln_fp8_te(*args, meta=kwargs["meta"])
    else:
        raise ValueError(f"Implementation '{impl}' is not recognized.")

    return result

def test(*args, **kwargs) -> torch.Tensor:
    impl = kwargs["impl"]
    warmup_iters = kwargs["warmup_iters"]
    main_iters = kwargs["main_iters"]

    print(f"Running {warmup_iters} warmup iters for {impl}...")
    for i in range(warmup_iters):
        result = bias_add_ln_fp8(*args, **kwargs)

    torch.cuda.profiler.start()
    print(f"Running {main_iters} main iters for {impl}...")
    for i in range(main_iters):
        result = bias_add_ln_fp8(*args, **kwargs)
    torch.cuda.profiler.stop()

    return result

def main():
    # Settings - BF16 input, FP8 output
    input_dtype = torch.bfloat16
    output_dtype = torch.float8_e4m3fn
    eps = 1e-5
    fp8_scale = 0.5
    x_shape = (512, 1, 12288)
    w_shape = (x_shape[-1],)
    warmup_iters = 10
    main_iters = 1000

    # LayerNorm params
    ln_weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    ln_bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)

    # Inputs
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    bias = torch.randn(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    residual = torch.randn_like(x)

    result_dict = {}
    #for impl in ["native", "compile", "te"]:
    for impl in ["compile"]:
        kwargs = {"impl": impl,
                  "warmup_iters": warmup_iters,
                  "main_iters": main_iters,
                 }
        if impl == "te":
            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
            meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / fp8_scale
            meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
            kwargs["meta"] = meta

        print(f"Testing {impl}...")
        result = test(x.detach().clone(),
                      bias.detach().clone(),
                      residual.detach().clone(),
                      w_shape,
                      ln_weight.detach().clone(),
                      ln_bias.detach().clone(),
                      eps,
                      fp8_scale,
                      output_dtype,
                      **kwargs)
        result_dict[impl] = result

        #print("*** " + impl + " ***")
        #print(result_dict[impl])

if __name__ == '__main__':
    main()
