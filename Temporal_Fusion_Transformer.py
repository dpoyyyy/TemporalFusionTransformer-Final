"""
Tft.py

This script contains a complete implementation of the Temporal Fusion Transformer (TFT)
model in PyTorch, designed for multi-horizon time series forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Tuple

# ======================================================================================
# SECTION 1: HELPER FUNCTIONS (Sparsemax, GLU)
# ======================================================================================

def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    A complete and numerically stable implementation of the Sparsemax function.

    Sparsemax projects the input onto the probability simplex, but unlike softmax,
    it can produce sparse probability distributions with exact zeros.
    """
    if input.dim() == 0:
        return input

    input_t = input.transpose(dim, -1)
    zs = torch.sort(input_t, dim=-1, descending=True)[0]

    k_range = torch.arange(1, zs.size(-1) + 1, device=zs.device, dtype=zs.dtype)
    view_shape = [1] * (zs.dim() - 1) + [zs.size(-1)]
    k_tensor = k_range.view(view_shape)

    # Find the support of the sparsemax projection
    bound = 1 + k_tensor * zs
    cumsum_zs = torch.cumsum(zs, dim=-1)
    is_gt = (bound > cumsum_zs).to(zs.dtype)
    k = torch.sum(is_gt, dim=-1, keepdim=True)

    k_safe = torch.clamp(k, min=1.0)

    # Compute the threshold tau
    zs_sparse = zs * is_gt
    tau = (torch.sum(zs_sparse, dim=-1, keepdim=True) - 1) / k_safe

    output = torch.clamp(input_t - tau, min=0.0)
    return output.transpose(dim, -1)

class GLUUnit(nn.Module):
    """Gated Linear Unit (GLU) helper module."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)

# ======================================================================================
# SECTION 2: CORE TFT MODULES
# ======================================================================================

class GRN(nn.Module):
    """
    Gated Residual Network (GRN) - A faithful implementation from the paper with Dropout.

    The GRN provides a flexible way to learn complex non-linear relationships and includes
    a residual connection to prevent vanishing gradients.
    """
    def __init__(self, d_input: int, d_model: int, d_context: Optional[int] = None, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.linear_a = nn.Linear(d_input, d_model)
        self.linear_c = nn.Linear(d_context, d_model, bias=False) if d_context is not None else None

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.glu = GLUUnit(d_model, d_model)

        self.proj_skip = nn.Linear(d_input, d_model) if d_input != d_model else None
        self.layer_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, a: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.proj_skip(a) if self.proj_skip is not None else a

        a_proj = self.linear_a(a)
        c_proj = self.linear_c(c) if (c is not None and self.linear_c is not None) else None

        if c_proj is not None and c_proj.dim() == 2 and a_proj.dim() == 3:
            c_proj = c_proj.unsqueeze(1).expand(-1, a_proj.shape[1], -1)

        combined = a_proj + (c_proj if c_proj is not None else 0)

        activated = self.activation(combined)
        dropped_out = self.dropout(activated)
        gated = self.glu(dropped_out)

        return self.layer_norm(residual + gated)

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - Final and refined version.

    This network learns to select or weight the most important input variables at each
    time step, enhancing interpretability and performance.
    """
    def __init__(self,
                 d_model: int,
                 input_dims: Dict[str, int],
                 var_types: Dict[str, str],
                 context_dim: Optional[int] = None,
                 selection_method: str = 'softmax'):
        super().__init__()
        self.variable_names = list(input_dims.keys())
        self.num_vars = len(self.variable_names)
        self.d_model = d_model
        self.var_types = var_types
        self.selection_method = selection_method

        # Input converters for categorical and continuous variables
        self.input_converters = nn.ModuleDict()
        for name, dim in input_dims.items():
            v_type = self.var_types.get(name, 'continuous')
            if v_type == 'categorical':
                self.input_converters[name] = nn.Embedding(dim, d_model)
            else:
                self.input_converters[name] = nn.Linear(dim, d_model)

        # A separate GRN for each variable
        self.variable_grns = nn.ModuleDict({
            name: GRN(d_model, d_model, d_context=context_dim)
            for name in self.variable_names
        })

        self.scoring_grn = GRN(self.num_vars * d_model, max(d_model, 32), d_context=context_dim)
        self.score_projection = nn.Linear(max(d_model, 32), self.num_vars)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, variables: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        is_time_varying = any(v.dim() == 3 for v in variables.values())
        T = next((v.shape[1] for v in variables.values() if v.dim() == 3), 1)

        raw_embs, processed = [], []
        for name in self.variable_names:
            v = variables[name]
            v_type = self.var_types.get(name, 'continuous')
            emb = self.input_converters[name](v.long() if v_type == 'categorical' else v)

            if is_time_varying:
                if emb.dim() == 2:
                    emb = emb.unsqueeze(1).expand(-1, T, -1)
                elif emb.dim() != 3:
                    raise ValueError(f"Unexpected embedding dimension for time-varying input '{name}': {emb.dim()}")
            else: # Static mode
                if emb.dim() == 3:
                    emb = emb.mean(dim=1)

            raw_embs.append(emb)
            processed.append(self.variable_grns[name](emb, context))

        # Calculate variable weights
        flattened = torch.cat(raw_embs, dim=-1)
        score_hidden = self.scoring_grn(flattened, context)
        scores = self.score_projection(score_hidden)
        weights = sparsemax(scores, dim=-1) if self.selection_method == 'sparsemax' else F.softmax(scores, dim=-1)

        # Apply weights to the processed variables
        stacked = torch.stack(processed, dim=-2)
        combined = torch.einsum('...vd,...v->...d', stacked, weights)
        return combined, weights

class StaticCovariateEncoder(nn.Module):
    """
    Static Covariate Encoder (SCE).

    This module uses GRNs to encode the static metadata into four distinct context vectors
    (c_s, c_c, c_h, c_e) that are used throughout the TFT model.
    """
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.context_encoders = nn.ModuleDict({
            n: GRN(d_model, d_model, dropout_rate=dropout_rate) for n in ['c_s', 'c_c', 'c_h', 'c_e']
        })
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {n: enc(x) for n, enc in self.context_encoders.items()}

class Seq2SeqLSTM(nn.Module):
    """
    Local processing layer using a sequence-to-sequence LSTM architecture.

    This module processes historical and future known inputs to capture local temporal patterns.
    """
    def __init__(self, d_model: int, hidden: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.encoder = nn.LSTM(d_model, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(d_model, hidden, num_layers=num_layers, batch_first=True)

        self.init_h_proj = nn.Linear(d_model, num_layers * hidden * (2 if bidirectional else 1))
        self.init_c_proj = nn.Linear(d_model, num_layers * hidden * (2 if bidirectional else 1))
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden = hidden

        if self.bidirectional:
            self.h_comb_proj = nn.Linear(2 * hidden, hidden)
            self.c_comb_proj = nn.Linear(2 * hidden, hidden)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _project_states(self, c_h_in: torch.Tensor, c_c_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects static context vectors to initial LSTM hidden and cell states."""
        B = c_h_in.shape[0]
        total_dims = self.num_layers * (2 if self.bidirectional else 1)
        h0 = self.init_h_proj(c_h_in).view(B, total_dims, self.hidden).permute(1, 0, 2).contiguous()
        c0 = self.init_c_proj(c_c_in).view(B, total_dims, self.hidden).permute(1, 0, 2).contiguous()
        return h0, c0

    def forward(self, hist: torch.Tensor, fut: torch.Tensor, c_h: torch.Tensor, c_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = hist.shape[0]
        h0_enc, c0_enc = self._project_states(c_h, c_c)
        enc_out, (h_n, c_n) = self.encoder(hist, (h0_enc, c0_enc))

        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, B, self.hidden).permute(0, 2, 1, 3).reshape(self.num_layers, B, 2 * self.hidden)
            c_n = c_n.view(self.num_layers, 2, B, self.hidden).permute(0, 2, 1, 3).reshape(self.num_layers, B, 2 * self.hidden)
            h_dec_init = self.h_comb_proj(h_n)
            c_dec_init = self.c_comb_proj(c_n)
        else:
            h_dec_init, c_dec_init = h_n, c_n

        dec_out, _ = self.decoder(fut, (h_dec_init, c_dec_init))
        return enc_out, dec_out

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention mechanism.

    This is a modified version of standard multi-head attention where attention weights
    are shared across heads for the value projection, improving interpretability.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V_shared = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            if mask.dim() == 2:
                mask_expanded = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask_expanded = mask.unsqueeze(1)
            else:
                mask_expanded = mask
            scores = scores.masked_fill(mask_expanded, float('-1e9'))

        attn_per_head = F.softmax(scores, dim=-1)
        attn_per_head = self.dropout(attn_per_head)

        # Aggregate attention weights across heads
        alpha_agg = attn_per_head.mean(dim=1)

        out = self.out_proj(torch.matmul(alpha_agg, V_shared))
        return out, alpha_agg

class TemporalProcessor(nn.Module):
    """
    Complete Temporal Processor for the TFT.

    This module integrates the local processing (LSTM) with the global processing
    (Interpretable Attention) and static context enrichment.
    """
    def __init__(self, d_model: int, hidden: int, num_layers: int = 1, n_heads: int = 4, dropout_rate: float = 0.1, bidirectional_encoder: bool = False):
        super().__init__()
        self.seq2seq = Seq2SeqLSTM(d_model, hidden, num_layers=num_layers, bidirectional=bidirectional_encoder)

        lstm_out_dim = hidden * (2 if bidirectional_encoder else 1)
        self.lstm_proj = nn.Linear(lstm_out_dim, d_model) if lstm_out_dim != d_model else None

        self.static_enrich = GRN(d_model, d_model, d_context=d_model, dropout_rate=dropout_rate)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.attention_gate = GRN(d_model, d_model, d_context=d_model, dropout_rate=dropout_rate)
        self.ffn_grn = GRN(d_model, d_model, dropout_rate=dropout_rate)
        self.final_gate = GRN(d_model, d_model, d_context=d_model, dropout_rate=dropout_rate)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        if self.lstm_proj: nn.init.xavier_uniform_(self.lstm_proj.weight)

    def forward(self, hist_emb: torch.Tensor, fut_emb: torch.Tensor, contexts: Dict[str, torch.Tensor], causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Local Processing
        enc_out, dec_out = self.seq2seq(hist_emb, fut_emb, contexts['c_h'], contexts['c_c'])
        if self.lstm_proj:
            enc_out = self.lstm_proj(enc_out)
        seq = torch.cat([enc_out, dec_out], dim=1)

        # 2. Static Enrichment
        enriched = self.static_enrich(seq, c=contexts['c_e'])
        residual = enriched

        # 3. Global Processing (Self-Attention)
        attn_out, alpha = self.attention(enriched, mask=causal_mask)
        attn_gated = self.attention_gate(attn_out, c=enriched)

        # 4. Position-wise Feed-Forward
        ffn_out = self.ffn_grn(attn_gated)

        # 5. Final Gating layer
        final_out = self.final_gate(residual, c=ffn_out)
        return final_out, alpha

# ======================================================================================
# SECTION 4: FINAL INTEGRATED TFT MODEL CLASS
# ======================================================================================

class TemporalFusionTransformer(nn.Module):
    """
    The complete and integrated Temporal Fusion Transformer model.

    This class brings together all the components: Variable Selection, Static Covariate
    Encoding, and the Temporal Processor to make multi-horizon forecasts.
    """
    def __init__(self,
                 d_model: int,
                 d_lstm_hidden: int,
                 static_input_dims: Dict[str, int],
                 static_var_types: Dict[str, str],
                 past_input_dims: Dict[str, int],
                 past_var_types: Dict[str, str],
                 future_input_dims: Dict[str, int],
                 future_var_types: Dict[str, str],
                 num_lstm_layers: int = 1,
                 num_attn_heads: int = 4,
                 dropout_rate: float = 0.1,
                 selection_method: str = 'softmax',
                 bidirectional_encoder: bool = False,
                 num_quantiles: int = 3):
        super().__init__()

        self.static_vsn = VariableSelectionNetwork(d_model, static_input_dims, static_var_types, selection_method=selection_method)
        self.past_vsn = VariableSelectionNetwork(d_model, past_input_dims, past_var_types, context_dim=d_model, selection_method=selection_method)
        self.future_vsn = VariableSelectionNetwork(d_model, future_input_dims, future_var_types, context_dim=d_model, selection_method=selection_method)
        self.static_encoder = StaticCovariateEncoder(d_model, dropout_rate)
        self.temporal_processor = TemporalProcessor(d_model, d_lstm_hidden, num_layers=num_lstm_layers, n_heads=num_attn_heads,
                                                  dropout_rate=dropout_rate, bidirectional_encoder=bidirectional_encoder)
        self.output_projection = nn.Linear(d_model, num_quantiles)

    def forward(self,
                static_vars: Dict[str, torch.Tensor],
                past_vars: Dict[str, torch.Tensor],
                future_vars: Dict[str, torch.Tensor],
                return_interpretation: bool = False):

        # 1. Process static variables and generate contexts
        static_combined, static_weights = self.static_vsn(static_vars)
        contexts = self.static_encoder(static_combined)

        # 2. Process time-varying variables
        past_emb, past_weights = self.past_vsn(past_vars, context=contexts['c_s'])
        future_emb, future_weights = self.future_vsn(future_vars, context=contexts['c_s'])

        # 3. Create causal mask for attention
        past_len = next(iter(past_vars.values())).shape[1]
        future_len = next(iter(future_vars.values())).shape[1]
        total_len = past_len + future_len
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=past_emb.device), diagonal=1).bool()

        # 4. Run the main temporal processing block
        temporal_output, attn_weights = self.temporal_processor(hist_emb=past_emb, fut_emb=future_emb,
                                                                contexts=contexts, causal_mask=causal_mask)

        # 5. Select future time steps for prediction
        prediction_input = temporal_output[:, past_len:, :]
        predictions = self.output_projection(prediction_input)

        if return_interpretation:
            interpretation = {
                "static_vsn_weights": static_weights,
                "past_vsn_weights": past_weights,
                "future_vsn_weights": future_weights,
                "attention_weights": attn_weights
            }
            return predictions, interpretation

        return predictions