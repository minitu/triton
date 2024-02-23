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

class _LayerNormFP8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: torch.float32,
        zero_centered_gamma: bool,
        meta: tex.FP8TensorMeta,
    ):
        ln_out = torch.empty_like(inp, dtype=torch.uint8)
        ln_out_kwarg = {"ln_out": ln_out}
        ln_out, mu, rsigma = tex.layernorm_fwd_fp8(inp,
                                             ln_weight,
                                             ln_bias,
                                             eps,
                                             meta,
                                             FP8_TENSOR_INDEX, # tex.FP8FwdTensors.GEMM1_INPUT
                                             tex.DType.kFloat8E4M3, # fp8_dtype_forward
                                             0, # TODO: fwd_ln_sm_margin
                                             zero_centered_gamma,
                                             **ln_out_kwarg,
        )

        ctx.save_for_backward(
            inp,
            ln_weight,
            mu,
            rsigma
        )
        ctx.zero_centered_gamma = zero_centered_gamma

        return ln_out

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        (inp, ln_weight, mu, rsigma) = ctx.saved_tensors

        dgrad, dgamma, dbeta = tex.layernorm_bwd(
            grad_output, inp, mu, rsigma, ln_weight, 0, ctx.zero_centered_gamma
        )

        return (
            dgrad,
            dgamma,
            dbeta)

class BiasAddLayerNormFP8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
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
        meta: tex.FP8TensorMeta,
    ):
        bda_out = bias_dropout_add_fused(x, bias, residual, 0, True)
        bda_out = bda_out.view(-1, w_shape[-1])

        ln_out = _LayerNormFP8.apply(bda_out, ln_weight, ln_bias, eps,
                                     zero_centered_gamma, meta)

        return bda_out, ln_out.view(output_dtype), meta.amax_history[0][FP8_TENSOR_INDEX]

bias_add_ln_fp8_te = BiasAddLayerNormFP8()

def bias_add_ln_fp8(*args, **kwargs) -> torch.Tensor:
    impl = kwargs["impl"]
    meta = kwargs["meta"]

    if impl == "native":
        result = bias_add_ln_fp8_native(*args, meta.scale[FP8_TENSOR_INDEX], meta.amax_history[0][FP8_TENSOR_INDEX])
    elif impl == "compile":
        result = bias_add_ln_fp8_compile(*args, meta.scale[FP8_TENSOR_INDEX], meta.amax_history[0][FP8_TENSOR_INDEX])
    elif impl == "te":
        result = bias_add_ln_fp8_te(*args, meta=meta)
    else:
        raise ValueError(f"Implementation '{impl}' is not recognized.")

    return result

def test(*args, **kwargs) -> torch.Tensor:
    impl = kwargs["impl"]
    warmup_iters = kwargs["warmup_iters"]
    main_iters = kwargs["main_iters"]
    do_bprop = kwargs["do_bprop"]

    loss = torch.nn.MSELoss()

    print(f"Running {warmup_iters} warmup iters for {impl}...")
    for i in range(warmup_iters):
        result = bias_add_ln_fp8(*args, **kwargs)
        if do_bprop:
            y = result[0].sum()
            y_ = torch.zeros_like(y)
            diff = loss(y, y_)
            diff.backward(retain_graph=True)

    torch.cuda.profiler.start()
    print(f"Running {main_iters} main iters for {impl}...")
    for i in range(main_iters):
        torch.cuda.nvtx.range_push("fprop")
        result = bias_add_ln_fp8(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        if do_bprop:
            y = result[0].sum()
            y_ = torch.zeros_like(y)
            diff = loss(y, y_)
            torch.cuda.nvtx.range_push("bprop")
            diff.backward(retain_graph=True)
            torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()

    return result

def main():
    # Implementations
    all_impls = ["native", "compile", "te"]

    # Settings - BF16 input, FP8 output
    input_dtype = torch.bfloat16
    output_dtype = torch.float8_e4m3fn
    eps = 1e-5
    fp8_scale = 0.5
    x_shape = (512, 1, 12288)
    w_shape = (x_shape[-1],)
    zero_centered_gamma = True
    warmup_iters = 10
    main_iters = 100
    do_bprop = True
    compare = True

    # LayerNorm params
    ln_weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    ln_bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)

    # Inputs
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    bias = torch.randn(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    residual = torch.randn_like(x)

    fprop_result = {}
    bprop_result = {}
    for impl in all_impls:
    #for impl in ["compile"]:
        meta = tex.FP8TensorMeta()
        meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
        meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / fp8_scale # TODO: Where is this used?
        meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")

        kwargs = {"impl": impl,
                  "warmup_iters": warmup_iters,
                  "main_iters": main_iters,
                  "meta": meta,
                  "do_bprop": do_bprop,
                 }

        print(f"Testing {impl}...")
        x_clone = x.detach().clone().requires_grad_()
        bias_clone = bias.detach().clone().requires_grad_()
        residual_clone = residual.detach().clone().requires_grad_()
        ln_weight_clone = ln_weight.detach().clone().requires_grad_()
        ln_bias_clone = ln_bias.detach().clone().requires_grad_()
        result = test(x_clone,
                      bias_clone,
                      residual_clone,
                      w_shape,
                      ln_weight_clone,
                      ln_bias_clone,
                      eps,
                      fp8_scale,
                      output_dtype,
                      zero_centered_gamma,
                      **kwargs)
        fprop_result[impl] = result
        bprop_result[impl] = {
                "x.wgrad": x_clone.grad,
                "bias.wgrad": bias_clone.grad,
                "residual.wgrad": residual_clone.grad,
                "ln_weight.wgrad": ln_weight_clone.grad,
                "ln_bias.wgrad": ln_bias_clone.grad
        }

    if compare:
        print("\nComparing fprop...\n")
        for i in range(len(all_impls)):
            for j in range(i+1, len(all_impls)):
                impl_cur = all_impls[i]
                impl_ref = all_impls[j]
                print("[" + impl_cur + "] vs. [" + impl_ref + "]")
                bda_out_diff = torch.amax(torch.abs(fprop_result[impl_cur][0] - fprop_result[impl_ref][0]))
                ln_out_diff = torch.amax(torch.abs(fprop_result[impl_cur][1].to(torch.bfloat16) - fprop_result[impl_ref][1].to(torch.bfloat16)))
                amax_diff = torch.abs(fprop_result[impl_cur][2] - fprop_result[impl_ref][2])
                print("Intermediate Bias-Dropout-Add output diff:", bda_out_diff.item())
                print("LayerNorm-FP8_Cast output diff (FP8 cast to BF16 for comparison):", ln_out_diff.item())
                print("Amax diff:", amax_diff.item())
                print()

        if do_bprop:
            print("Comparing bprop...\n")
            for i in range(len(all_impls)):
                for j in range(i+1, len(all_impls)):
                    impl_cur = all_impls[i]
                    impl_ref = all_impls[j]
                    print("[" + impl_cur + "] vs. [" + impl_ref + "]")
                    for key in bprop_result[impl_cur].keys():
                        has_none = False
                        if bprop_result[impl_cur][key] is None:
                            print(f"{impl_cur} has None for {key}")
                            has_none = True
                        if bprop_result[impl_ref][key] is None:
                            print(f"{impl_ref} has None for {key}")
                            has_none = True
                        if not has_none:
                            print(key + ":", torch.amax(torch.abs(bprop_result[impl_cur][key] - bprop_result[impl_ref][key])))
                    print()

if __name__ == '__main__':
    main()
