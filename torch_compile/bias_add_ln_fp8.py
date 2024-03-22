import torch
import transformer_engine.pytorch.cpp_extensions as tex
import nvfuser
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype

FP8_TENSOR_INDEX = 0

# ------------------------------ Native ------------------------------
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

# ------------------------------ Compile ------------------------------
@torch.compile
def bias_add_ln_fp8_compile(*args) -> torch.Tensor:
    return bias_add_ln_fp8_native(*args)

# ------------------------------ TE ------------------------------
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

class _LayerNormFP8TE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: torch.float32,
        zero_centered_gamma: bool,
        output_dtype: torch.dtype,
        meta: tex.FP8TensorMeta,
        do_bprop: bool
    ):
        torch.cuda.nvtx.range_push("TE forward empty_like")
        ln_out = torch.empty_like(inp, dtype=torch.uint8)
        torch.cuda.nvtx.range_pop()
        output_kwarg = {"ln_out": ln_out}
        torch.cuda.nvtx.range_push("TE layernorm_fwd_fp8")
        ln_out, mu, rsigma = tex.layernorm_fwd_fp8(inp,
                                                   ln_weight,
                                                   ln_bias,
                                                   eps,
                                                   meta,
                                                   FP8_TENSOR_INDEX, # tex.FP8FwdTensors.GEMM1_INPUT
                                                   tex.DType.kFloat8E4M3, # fp8_dtype_forward
                                                   0, # TODO: fwd_ln_sm_margin
                                                   zero_centered_gamma,
                                                   **output_kwarg,
        )
        torch.cuda.nvtx.range_pop()

        if do_bprop:
            # Cast LN output to BF16 for bprop, need to remove from perf cost
            #ln_out = ln_out.to(torch.bfloat16)
            torch.cuda.nvtx.range_push("TE forward view, requires_grad")
            ln_out = ln_out.view(output_dtype)
            ln_out.requires_grad_()
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("TE forward save")
            ctx.save_for_backward(
                inp,
                ln_weight,
                mu,
                rsigma
            )
            ctx.zero_centered_gamma = zero_centered_gamma
            torch.cuda.nvtx.range_pop()

        return ln_out

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        (inp, ln_weight, mu, rsigma) = ctx.saved_tensors

        # Incoming grad cast to BF16, need to remove from perf cost
        torch.cuda.nvtx.range_push("TE backward cast")
        grad_output_bf16 = grad_output.to(torch.bfloat16)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("TE layernorm_bwd")
        dgrad, dgamma, dbeta = tex.layernorm_bwd(
            grad_output_bf16, inp, mu, rsigma, ln_weight, 0, ctx.zero_centered_gamma
        )
        torch.cuda.nvtx.range_pop()

        return (
            dgrad,
            dgamma,
            dbeta,
            None,
            None,
            None,
            None,
            None,
        )

class BiasAddLayerNormFP8TE(torch.nn.Module):
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
        do_bprop: bool,
    ):
        bda_out = bias_dropout_add_fused(x, bias, residual, 0, True)
        bda_out = bda_out.view(-1, w_shape[-1])

        ln_out = _LayerNormFP8TE.apply(bda_out, ln_weight, ln_bias, eps,
                                       zero_centered_gamma, output_dtype, meta, do_bprop)
        #ln_out_ret = ln_out.to(output_dtype) if do_bprop else ln_out.view(output_dtype)
        ln_out_ret = ln_out if do_bprop else ln_out.view(output_dtype)

        return bda_out, ln_out_ret, meta.amax_history[0][FP8_TENSOR_INDEX]

bias_add_ln_fp8_te = BiasAddLayerNormFP8TE()

# ------------------------------ TE Fused ------------------------------
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

class _BiasAddLayerNormFP8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        bda_bias: torch.Tensor,
        bda_residual: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: torch.float32,
        zero_centered_gamma: bool,
        output_dtype: torch.dtype,
        meta: tex.FP8TensorMeta,
        do_bprop: bool
    ):
        torch.cuda.nvtx.range_push("TE forward empty_like")
        bda_out = torch.empty_like(inp, dtype=inp.dtype)
        ln_out = torch.empty_like(inp, dtype=torch.uint8)
        torch.cuda.nvtx.range_pop()
        output_kwarg = {"bda_out": bda_out, "ln_out": ln_out}
        torch.cuda.nvtx.range_push("TE bias_add_layernorm_fwd_fp8")
        bda_out, ln_out, mu, rsigma = tex.bias_add_layernorm_fwd_fp8(inp,
                                                                     bda_bias,
                                                                     bda_residual,
                                                                     ln_weight,
                                                                     ln_bias,
                                                                     eps,
                                                                     meta,
                                                                     FP8_TENSOR_INDEX, # tex.FP8FwdTensors.GEMM1_INPUT
                                                                     tex.DType.kFloat8E4M3, # fp8_dtype_forward
                                                                     0, # TODO: fwd_ln_sm_margin
                                                                     zero_centered_gamma,
                                                                     **output_kwarg,
        )
        torch.cuda.nvtx.range_pop()

        # TODO
        if do_bprop:
            # Cast LN output to BF16 for bprop, need to remove from perf cost
            #ln_out = ln_out.to(torch.bfloat16)
            torch.cuda.nvtx.range_push("TE forward view, requires_grad")
            ln_out = ln_out.view(output_dtype)
            ln_out.requires_grad_()
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("TE forward save")
            ctx.save_for_backward(
                inp,
                ln_weight,
                mu,
                rsigma
            )
            ctx.zero_centered_gamma = zero_centered_gamma
            torch.cuda.nvtx.range_pop()

        return bda_out, ln_out

    # TODO
    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        (inp, ln_weight, mu, rsigma) = ctx.saved_tensors

        # Incoming grad cast to BF16, need to remove from perf cost
        torch.cuda.nvtx.range_push("TE backward cast")
        grad_output_bf16 = grad_output.to(torch.bfloat16)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("TE layernorm_bwd")
        dgrad, dgamma, dbeta = tex.layernorm_bwd(
            grad_output_bf16, inp, mu, rsigma, ln_weight, 0, ctx.zero_centered_gamma
        )
        torch.cuda.nvtx.range_pop()

        return (
            dgrad,
            dgamma,
            dbeta,
            None,
            None,
            None,
            None,
            None,
        )

class BiasAddLayerNormFP8Fused(torch.nn.Module):
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
        do_bprop: bool,
    ):
        # TODO: Remove views?
        bda_out, ln_out = _BiasAddLayerNormFP8.apply(x.view(-1, w_shape[-1]), bias, residual.view(-1, w_shape[-1]),
                                                     ln_weight, ln_bias, eps,
                                                     zero_centered_gamma, output_dtype, meta, do_bprop)
        ln_out_ret = ln_out.view(output_dtype)

        return bda_out, ln_out_ret, meta.amax_history[0][FP8_TENSOR_INDEX]

bias_add_ln_fp8_te_fused = BiasAddLayerNormFP8Fused()

# ------------------------------ nvFuser ------------------------------
def partially_contig_tensor(
    fd: "nvfuser.FusionDefinition",
    x: torch.Tensor,
) -> "nvfuser.Tensor":
    return fd.define_tensor(
        sizes=x.size(),
        strides=x.stride(),
        dtype=torch_dtype_to_nvfuser_dtype(x.dtype),
    )

class _BiasAddLayerNormFuser(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
        w_shape: tuple,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: torch.float32,
        output_dtype: torch.dtype,
        zero_centered_gamma: bool,
        scale_tensor: torch.Tensor,
        amax_tensor: torch.Tensor,
    ):
        with nvfuser.FusionDefinition() as fd:
            x_f = partially_contig_tensor(fd, x)
            bias_f = partially_contig_tensor(fd, bias)
            residual_f = partially_contig_tensor(fd, residual)
            ln_weight_f = partially_contig_tensor(fd, ln_weight)
            ln_bias_f = partially_contig_tensor(fd, ln_bias)
            scale_f = partially_contig_tensor(fd, scale_tensor)
            amax_f = partially_contig_tensor(fd, amax_tensor)

            # Add 1 to LayerNorm weights if needed
            if zero_centered_gamma:
                G0 = fd.define_scalar(1, dtype=nvfuser.DataType.Int)
                ln_weight_f = fd.ops.add(ln_weight_f, G0)

            # Bias-Add
            V17 = x_f.shape()
            B0 = fd.ops.broadcast_in_dim(bias_f, shape=V17, broadcast_dims=[1])
            T0 = fd.ops.add(x_f, B0)
            T1 = fd.ops.add(T0, residual_f)
            # Bias-Add can output as bfloat16
            T1_half = fd.ops.cast(T1, nvfuser.DataType.BFloat16)
            fd.add_output(T1_half) # Bias-Add output

            # LayerNorm
            T2, T3 = fd.ops.var_mean(T1, [1], correction=0, keepdim=False)

            V6 = fd.define_vector([T1.size(0), 1], dtype=nvfuser.DataType.Int)
            T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[0])
            T11 = fd.ops.broadcast_in_dim(T3, shape=V6, broadcast_dims=[0])

            S12 = fd.define_scalar(eps, dtype=nvfuser.DataType.Double)
            T13 = fd.ops.add(T7, S12)
            T14 = fd.ops.rsqrt(T13)

            T18 = fd.ops.broadcast_in_dim(T11, shape=V17, broadcast_dims=[0, 1])
            T19 = fd.ops.sub(T1, T18)
            T23 = fd.ops.broadcast_in_dim(T14, shape=V17, broadcast_dims=[0, 1])
            T24 = fd.ops.mul(T19, T23)

            T25 = fd.ops.broadcast_in_dim(ln_weight_f, shape=V17, broadcast_dims=[1])
            T26 = fd.ops.mul(T24, T25)
            T27 = fd.ops.broadcast_in_dim(ln_bias_f, shape=V17, broadcast_dims=[1])
            T28 = fd.ops.add(T26, T27)

            # Amax
            # first half of amax is done along with the layer_norm kernel
            T29 = fd.ops.abs(T28)
            T29_sum = fd.ops.max(T29, [1])

            # NOTE: manual segmentation / reduced precision on intermediate
            # doesn't seem to save any kernel time
            #T29_sum_fp8 = fd.ops.cast(T29_sum, dtype=nvfuser.DataType.BFloat16)
            ## manaul segmentation
            #T_seg_2 = fd.ops.segment_set(T29_sum_fp8)
            ## second half of amax is done in a standalone kernel
            #T29_sum_fp32 = fd.ops.cast(T_seg_2, dtype=torch_dtype_to_nvfuser_dtype(torch.float32))
            #T30 = fd.ops.max(T29_sum_fp32 )

            # second half of amax is done in a standalone kernel
            T30 = fd.ops.max(T29_sum )

            # Scale
            T31 = fd.ops.mul(T28, scale_f)

            T32 = fd.ops.cast(T31, dtype=torch_dtype_to_nvfuser_dtype(output_dtype))
            fd.add_output(T32) # LayerNorm output
            fd.add_output(T30, alias_input=amax_f) # Amax output inplace update

        bda_out, ln_out = fd.execute([x, bias, residual,
                                                   ln_weight, ln_bias,
                                                   scale_tensor, amax_tensor])

        return bda_out, ln_out

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        # TODO
        pass

class BiasAddLayerNormFuser(torch.nn.Module):
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
        scale_tensor: torch.Tensor,
        amax_tensor: torch.Tensor,
        do_bprop: bool,
    ):
        x = x.view(-1, w_shape[-1])
        residual = residual.view(-1, w_shape[-1])
        bda_out, ln_out = _BiasAddLayerNormFuser.apply(x, bias, residual, w_shape,
                                              ln_weight, ln_bias, eps,
                                              output_dtype, zero_centered_gamma,
                                              scale_tensor, amax_tensor)

        return bda_out, ln_out

bias_add_ln_fp8_nvfuser = BiasAddLayerNormFuser()

def bias_add_ln_fp8(*args, **kwargs) -> torch.Tensor:
    impl = kwargs["impl"]
    meta = kwargs["meta"]
    do_bprop = kwargs["do_bprop"]

    if impl == "native":
        result = bias_add_ln_fp8_native(*args, meta.scale[FP8_TENSOR_INDEX], meta.amax_history[0][FP8_TENSOR_INDEX])
    elif impl == "compile":
        result = bias_add_ln_fp8_compile(*args, meta.scale[FP8_TENSOR_INDEX], meta.amax_history[0][FP8_TENSOR_INDEX])
    elif impl == "te":
        result = bias_add_ln_fp8_te(*args, meta=meta, do_bprop=do_bprop)
    elif impl == "te_fused":
        result = bias_add_ln_fp8_te_fused(*args, meta=meta, do_bprop=do_bprop)
    elif impl == "nvfuser":
        result = bias_add_ln_fp8_nvfuser(*args, meta.scale[FP8_TENSOR_INDEX], meta.amax_history[0][FP8_TENSOR_INDEX], do_bprop)
    else:
        raise ValueError(f"Implementation '{impl}' is not recognized.")

    return result

def main():
    # Implementations
    all_impls = ["native", "compile", "te", "te_fused", "nvfuser"]

    # Settings - BF16 input, FP8 output
    input_dtype = torch.bfloat16
    output_dtype = torch.float8_e4m3fn
    eps = 1e-5
    fp8_scale = 0.5
    x_shape = (512, 1, 12288)
    w_shape = (x_shape[-1],)
    zero_centered_gamma = True
    warmup_iters = 0
    main_iters = 1
    do_bprop = False
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
    #for impl in ["nvfuser"]:
        ln_weight_impl = ln_weight.detach().clone().requires_grad_()
        ln_bias_impl = ln_bias.detach().clone().requires_grad_()
        x_impl = x.detach().clone().requires_grad_()
        bias_impl = bias.detach().clone().requires_grad_()
        residual_impl = residual.detach().clone().requires_grad_()

        kwargs = {"impl": impl,
                  "warmup_iters": warmup_iters,
                  "main_iters": main_iters,
                  "do_bprop": do_bprop,
                 }

        print(f"Testing {impl}...")
        print(f"Running {warmup_iters} warmup iters for {impl}...")
        for i in range(warmup_iters):
            x_clone = x_impl.detach().clone().requires_grad_()
            bias_clone = bias_impl.detach().clone().requires_grad_()
            residual_clone = residual_impl.detach().clone().requires_grad_()

            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
            meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / fp8_scale # TODO: Where is this used?
            meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
            kwargs["meta"] = meta

            result = bias_add_ln_fp8(
                x_clone,
                bias_clone,
                residual_clone,
                w_shape,
                ln_weight_impl,
                ln_bias_impl,
                eps,
                fp8_scale,
                output_dtype,
                zero_centered_gamma,
                **kwargs)

            if do_bprop:
                y = result[1].to(torch.bfloat16).mean()
                y.backward()

        print(f"Running {main_iters} main iters for {impl}...")
        torch.cuda.profiler.start()
        for i in range(main_iters):
            torch.cuda.nvtx.range_push("setup")

            x_clone = x_impl.detach().clone().requires_grad_()
            bias_clone = bias_impl.detach().clone().requires_grad_()
            residual_clone = residual_impl.detach().clone().requires_grad_()

            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
            meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / fp8_scale # TODO: Where is this used?
            meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
            kwargs["meta"] = meta

            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("fprop")

            result = bias_add_ln_fp8(
                x_clone,
                bias_clone,
                residual_clone,
                w_shape,
                ln_weight_impl,
                ln_bias_impl,
                eps,
                fp8_scale,
                output_dtype,
                zero_centered_gamma,
                **kwargs)

            torch.cuda.nvtx.range_pop()

            if do_bprop:
                torch.cuda.nvtx.range_push("loss")

                y = result[1].to(torch.bfloat16).mean()

                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("bprop")

                y.backward()

                torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()

        fprop_result[impl] = result
        bprop_result[impl] = {
                "x.wgrad": x_clone.grad,
                "bias.wgrad": bias_clone.grad,
                "residual.wgrad": residual_clone.grad,
                "ln_weight.wgrad": ln_weight_impl.grad,
                "ln_bias.wgrad": ln_bias_impl.grad
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
