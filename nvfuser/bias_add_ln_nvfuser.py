import torch
import nvfuser
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype

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
            #T1_half = fd.ops.cast(T1, nvfuser.DataType.BFloat16)
            #fd.add_output(T1_half) # Bias-Add output

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
            T29 = fd.ops.abs(T28)
            T29_sum = fd.ops.max(T29, [1])
            T29_sum_fp8 = fd.ops.cast(T29_sum, dtype=torch_dtype_to_nvfuser_dtype(output_dtype))
            T_seg_2 = fd.ops.segment_set(T29_sum_fp8)

            T29_sum_fp32 = fd.ops.cast(T_seg_2, dtype=torch_dtype_to_nvfuser_dtype(torch.float32))
            T30 = fd.ops.max(T29_sum_fp32 )

            # Scale
            T31 = fd.ops.mul(T28, scale_f)

            fd.add_output(T1) # Bias-Add output
            T32 = fd.ops.cast(T31, dtype=torch_dtype_to_nvfuser_dtype(output_dtype))
            fd.add_output(T32) # LayerNorm output
            fd.add_output(T30, alias_input=amax_f) # Amax output, TODO: fill input tensor instead?

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
        output_dtype: torch.dtype,
        zero_centered_gamma: bool,
        scale_tensor: torch.Tensor,
        amax_tensor: torch.Tensor,
    ):
        bda_out, ln_out = _BiasAddLayerNormFuser.apply(x, bias, residual, w_shape,
                                              ln_weight, ln_bias, eps,
                                              output_dtype, zero_centered_gamma,
                                              scale_tensor, amax_tensor)

        return bda_out, ln_out


def main():
    input_dtype = torch.bfloat16
    output_dtype = torch.float8_e4m3fn
    eps = 1e-5
    fp8_scale = 0.5
    x_shape = (512, 12288)
    w_shape = (x_shape[-1],)
    zero_centered_gamma = True

    # LayerNorm params
    ln_weight = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    ln_bias = torch.rand(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)

    # Inputs
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    bias = torch.randn(w_shape, dtype=input_dtype, device='cuda', requires_grad=True)
    residual = torch.randn_like(x)

    # FP8 metadata
    fp8_scale = torch.ones(1,dtype=torch.float32, device="cuda") * fp8_scale
    fp8_amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")

    # Create 'model' and do fprop
    bias_add_ln_nvfuser = BiasAddLayerNormFuser()
    print("amax before: ", fp8_amax_history)
    bda_out, ln_out = bias_add_ln_nvfuser(x, bias, residual, w_shape,
                                                       ln_weight, ln_bias, eps,
                                                       output_dtype, zero_centered_gamma,
                                                       fp8_scale[0], fp8_amax_history[0][0])
    print("amax after: ", fp8_amax_history)

if __name__ == '__main__':
    main()
