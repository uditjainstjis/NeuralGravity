import mlx.core as mx
import mlx.nn as nn
import math

class HybridLinear(nn.Module):
    """
    Implements the Hybrid Low-Rank Adaptation (HLRA) module for 2-bit quantized weights.
    Formulation from OCR'd doc:
    W_hybrid = m * (W_0 + B_dora @ A_dora) / ||W_0 + B_dora @ A_dora||_c + B_eora @ A_eora
    """
    def __init__(self, base_layer, rank=16, alpha=16.0, eora_rank=16, init_eora_with_svd=True):
        super().__init__()
        # Base layer should be the original quantized linear layer.
        self.base_layer = base_layer
        self.in_features = base_layer.weight.shape[1]
        self.out_features = base_layer.weight.shape[0]
        
        # DoRA Path parameters
        self.rank = rank
        self.scale = alpha / rank
        
        # A_dora: r x in_features
        self.A_dora = mx.random.normal((rank, self.in_features)) * math.sqrt(1 / self.in_features)
        # B_dora: out_features x r (initialized to 0)
        self.B_dora = mx.zeros((self.out_features, rank))
        
        # Magnitude vector m: initialize to the norm of the original weights
        # For quantized weights, we dequantize them once to get the norm
        W_0 = self._get_dequantized_weight()
        # norm computed across the input dimension (axis 1)
        m_init = mx.linalg.norm(W_0, axis=1, keepdims=True)
        self.m = m_init
        
        # EoRA Path parameters
        self.eora_rank = eora_rank
        self.A_eora = mx.zeros((eora_rank, self.in_features))
        self.B_eora = mx.zeros((self.out_features, eora_rank))
        
        if init_eora_with_svd:
            self._init_eora(W_0)

    def _get_dequantized_weight(self):
        # Depending on base_layer class, we extract the FP16/BF16 weights.
        # mlx.nn.QuantizedLinear provides `weight` as quantized, we need to explicitly dequantize.
        # Since mlx-lm usually wraps this, we can perform a forward pass with an identity.
        if hasattr(self.base_layer, "scales"): # It's a quantized layer
            # Dequantize: W_0 = dequantize(weight, scales, biases, group_size, bits)
            # MLX has mx.dequantize
            w_shape = (self.out_features, self.in_features)
            return mx.dequantize(
                self.base_layer.weight,
                self.base_layer.scales,
                self.base_layer.biases,
                self.base_layer.group_size,
                self.base_layer.bits
            ).reshape(w_shape)
        return self.base_layer.weight

    def _init_eora(self, W_0):
        """
        Initializes the EoRA matrices via Singular Value Decomposition of the 
        exact quantization error matrix: E = W_orig - Q(W_orig)
        """
        # If the base layer is a standard dense layer (FP16/BF16), we can compute 
        # the exact quantization error natively on the M3 GPU.
        if not hasattr(self.base_layer, "scales"):
            W_orig = self.base_layer.weight
            
            # 1. Simulate the aggressive 4-bit quantization natively in MLX
            W_quantized, scales, biases = mx.quantize(W_orig, group_size=64, bits=4)
            
            # 2. Dequantize back to standard manifold to calculate structural loss
            W_q = mx.dequantize(W_quantized, scales, biases, group_size=64, bits=4).reshape(W_orig.shape)
            
            # 3. Compute the True Quantization Error Matrix (E)
            E = W_orig - W_q
            
            orig_dtype = W_orig.dtype
            # 4. Perform SVD explicitly on the Error Matrix
            u, s, vh = mx.linalg.svd(E.astype(mx.float32), stream=mx.cpu)
            
            # 5. Extract lowest singular vectors targeting the lost/clipped high-frequency noise
            self.B_eora = u[:, -self.eora_rank:].astype(orig_dtype)
            self.A_eora = ((vh[-self.eora_rank:, :].T * s[-self.eora_rank:]).T).astype(orig_dtype)
            
        else:
            # Fallback for models that were force-loaded as Pre-Quantized lacking W_orig.
            # We fallback to lower-subspace approximation of the dequantized state.
            orig_dtype = W_0.dtype
            u, s, vh = mx.linalg.svd(W_0.astype(mx.float32), stream=mx.cpu)
            self.B_eora = u[:, -self.eora_rank:].astype(orig_dtype)
            self.A_eora = ((vh[-self.eora_rank:, :].T * s[-self.eora_rank:]).T).astype(orig_dtype)


    def _get_weight(self):
        # Calculate DoRA direction
        W_0 = self._get_dequantized_weight()
        dora_update = self.B_dora @ self.A_dora * self.scale
        directional_component = W_0 + dora_update
        
        # Normalize directional component
        col_norm = mx.linalg.norm(directional_component, axis=1, keepdims=True)
        # Avoid division by zero
        normalized_direction = directional_component / (col_norm + 1e-8)
        
        # DoRA Path Output
        W_dora = self.m * normalized_direction
        
        # EoRA Path Compensation
        eora_compensation = self.B_eora @ self.A_eora
        
        # Final Hybrid Weight
        W_hybrid = W_dora + eora_compensation
        return W_hybrid

    def __call__(self, x):
        # Forward pass: x @ W_hybrid.T + b
        W_hybrid = self._get_weight()
        out = x @ W_hybrid.T
        if "bias" in self.base_layer and self.base_layer.bias is not None:
            out += self.base_layer.bias
        return out
