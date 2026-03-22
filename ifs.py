# -*- coding: utf-8 -*-
"""
Execution Trace Embedding Module
=====================================
A complete Transformer-based embedding system for simple assembly execution traces.

Usage:
    embedder = TraceEmbedder()
    embedding = embedder.embed_trace(trace_lines)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast
import math
from generate_trace_pair import generate_trace_pair

# =============================================================================
# CUSTOM TOKENIZER FOR ARM ASSEMBLY
# =============================================================================

class AssemblyTokenizer:
    """
    Custom tokenizer for ARM assembly instructions.

    Memory addresses are normalised to ``SET_NNN`` tokens that encode the L2
    cache set index rather than the raw hex value.  This ensures that two
    traces with structurally identical interference patterns (same cache-set
    access sequences) produce the same token stream regardless of their
    absolute addresses, which dramatically improves generalisation.

    Normalisation formula (mirrors generate_trace_pair.py):
        line  = addr // LINE_SIZE
        set   = line % L2_SETS          # 256 sets
        token = f"SET_{set:03d}"        # e.g. "SET_031"
    """

    # Keep in sync with generate_trace_pair constants
    _LINE_SIZE = 32
    _L2_SETS   = 256

    def __init__(self, base_model_name: str = "microsoft/codebert-base"):
        """
        Initialize with a pre-trained code tokenizer as base.

        Args:
            base_model_name: HuggingFace model name for base tokenizer
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.pad_token_id = self.base_tokenizer.pad_token_id
        self.cls_token_id = self.base_tokenizer.cls_token_id
        self.sep_token_id = self.base_tokenizer.sep_token_id
        self.max_length = 512

        # Regex pattern to split assembly into meaningful tokens
        # Captures: Opcodes, Registers, Hex values, Decimals, Brackets, Commas
        self.token_pattern = re.compile(
            r'([A-Z]{2,4}|'           # Opcodes (LDR, STR, ADD, etc.)
            r'[Rr][0-9]{1,2}|'         # Registers (R0-R15, r0-r15)
            r'[Ss][Pp]|[Ll][Rr]|[Pp][Cc]|'  # Special regs (SP, LR, PC)
            r'#?0[xX][0-9A-Fa-f]+|'    # Hex values (#0x2000, 0x2000)
            r'#?[0-9]+|'               # Decimal values (#15, 15)
            r'\[|\]|,|#|-|\+)'         # Special characters
        )

    def _normalize_address(self, hex_token: str) -> str:
        """
        Convert a raw hex address token (e.g. ``0x1FE0``, ``#0x1FE0``) to a
        cache-set token (e.g. ``SET_031``).

        Non-address hex literals (small constants like ``#0x1``) that resolve
        to the same line as dozens of other instructions are left as ``HEX``
        so the model does not conflate them with memory addresses.  We use a
        simple heuristic: values >= LINE_SIZE are treated as addresses.
        """
        raw = hex_token.lstrip('#')
        try:
            value = int(raw, 16)
        except ValueError:
            return hex_token          # shouldn't happen given the regex, but safe

        if value < self._LINE_SIZE:
            return "HEX"              # tiny constant, not a meaningful address

        cache_set = (value // self._LINE_SIZE) % self._L2_SETS
        return f"SET_{cache_set:03d}"

    def tokenize_instruction(self, instruction: str) -> List[str]:
        """
        Tokenize a single assembly instruction.

        Hex address operands are normalised to ``SET_NNN`` tokens via
        ``_normalize_address`` so that structurally equivalent traces produce
        identical token sequences regardless of their absolute load addresses.

        Args:
            instruction: Assembly instruction string (e.g., "LDR R3, [0x1FE0]")

        Returns:
            List of token strings
        """
        # Remove leading/trailing whitespace
        instruction = instruction.strip()

        # Skip empty lines or comments
        if not instruction or instruction.startswith(';'):
            return []

        # Remove inline comments
        if ';' in instruction:
            instruction = instruction.split(';')[0].strip()

        # Remove address prefix if present (e.g., "0x0024 LDR R3, [R0]")
        parts = instruction.split()
        if len(parts) > 1 and re.match(r'^0x[0-9A-Fa-f]+$', parts[0]):
            instruction = ' '.join(parts[1:])

        # Split into sub-tokens using regex
        raw_tokens = self.token_pattern.findall(instruction)

        # Normalise hex address tokens → SET_NNN
        tokens = []
        hex_re = re.compile(r'^#?0[xX][0-9A-Fa-f]+$')
        for tok in raw_tokens:
            if hex_re.match(tok):
                tokens.append(self._normalize_address(tok))
            else:
                tokens.append(tok)

        return tokens

    def tokenize_trace(self, trace_lines: Union[str, List[str]],
                      max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize a complete execution trace.

        Args:
            trace_lines: Single string or list of instruction strings
            max_length: Maximum sequence length (default: self.max_length)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        if max_length is None:
            max_length = self.max_length

        # Convert to list if single string
        if isinstance(trace_lines, str):
            trace_lines = [line.strip() for line in trace_lines.split('\n') if line.strip()]

        # Tokenize all instructions
        all_tokens = []
        for line in trace_lines:
            instr_tokens = self.tokenize_instruction(line)
            if instr_tokens:
                all_tokens.extend(instr_tokens)
                all_tokens.append('[SEP]')  # Separate instructions

        # Remove trailing separator
        if all_tokens and all_tokens[-1] == '[SEP]':
            all_tokens = all_tokens[:-1]

        # Truncate if necessary
        if len(all_tokens) > max_length - 2:  # Reserve space for [CLS] and [SEP]
            all_tokens = all_tokens[:max_length - 2]

        # Add [CLS] and [SEP] tokens
        all_tokens = ['[CLS]'] + all_tokens + ['[SEP]']

        # Convert to input IDs using base tokenizer
        encoding = self.base_tokenizer(
            all_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.base_tokenizer.decode(input_ids, skip_special_tokens=True)


# =============================================================================
# ROTARY POSITIONAL EMBEDDINGS (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) for better sequence modeling.
    Superior to absolute positional encodings for capturing relative positions.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        """
        Initialize RoPE.

        Args:
            dim: Dimension of the embedding (must be even)
            max_seq_len: Maximum sequence length to support
            base: Base value for frequency calculation
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin tables
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        """Precompute cos and sin cache for efficiency."""
        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            seq_len: Actual sequence length (for caching)

        Returns:
            Tuple of (cos, sin) tensors for rotation
        """
        if seq_len is None:
            seq_len = x.shape[1]

        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine values from RoPE
        sin: Sine values from RoPE

    Returns:
        Tuple of rotated (q, k) tensors
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# ROPE-ENHANCED SELF-ATTENTION (post-encoder refinement layer)
# =============================================================================

class RoPEAttention(nn.Module):
    """
    A single multi-head self-attention layer that applies RoPE to Q and K.

    This is inserted *after* the frozen (or fine-tuned) CodeBERT encoder as a
    lightweight refinement layer, so we get proper relative-position encoding
    without needing to rewrite every internal attention block of CodeBERT.

    Shape contract: (batch, seq_len, dim) → (batch, seq_len, dim)
    """

    def __init__(self, dim: int, num_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:               (batch, seq_len, dim)
            attention_mask:  (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, dim)  residual-connected + layer-normed output
        """
        B, T, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        # Project and reshape to (B, H, T, HD)
        q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)

        # Apply RoPE to Q and K (cos/sin have shape (T, HD))
        cos, sin = self.rope(x, seq_len=T)
        # Expand to (1, 1, T, HD) for broadcasting across batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if attention_mask is not None:
            # Convert (B, T) mask to additive bias: 0 → 0.0, padding → -inf
            additive = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            attn = attn - additive * 1e9

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                         # (B, H, T, HD)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(out)

        # Pre-norm residual connection
        return self.norm(x + out)


# =============================================================================
# TRANSFORMER EMBEDDER WITH RoPE
# =============================================================================

class TraceEmbedder(nn.Module):
    """
    Transformer-based embedding model for ARM execution traces.

    Architecture
    ------------
    1. CodeBERT encoder   — contextual token representations
    2. RoPEAttention      — one post-encoder self-attention layer that applies
                            Rotary Positional Embeddings to Q and K, giving the
                            model genuine relative-position awareness
    3. Mean pooling       — collapse sequence → single vector
    4. Linear projection  — resize to target embedding_dim (if needed)
    5. LayerNorm + L2     — normalise for cosine-similarity training
    """

    def __init__(self,
                 model_name: str = "microsoft/codebert-base",
                 embedding_dim: int = 768,
                 max_seq_len: int = 512,
                 freeze_base: bool = False,
                 use_rope: bool = True):
        """
        Args:
            model_name:    HuggingFace model name for the base transformer
            embedding_dim: Output embedding dimension
            max_seq_len:   Maximum sequence length
            freeze_base:   Whether to freeze base CodeBERT weights
            use_rope:      Whether to insert the RoPEAttention refinement layer
        """
        super().__init__()

        # Load base transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.base_dim = self.transformer.config.hidden_size

        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # RoPE-enhanced post-encoder attention (properly wired this time)
        self.use_rope = use_rope
        if use_rope:
            self.rope_attn = RoPEAttention(
                dim=self.base_dim,
                num_heads=8,
                max_seq_len=max_seq_len,
            )

        # Optional projection to a different embedding dimension
        if embedding_dim != self.base_dim:
            self.projection = nn.Linear(self.base_dim, embedding_dim)
        else:
            self.projection = nn.Identity()

        self.embedding_dim = embedding_dim
        self.max_seq_len   = max_seq_len

        # Tokenizer (with address → cache-set normalisation)
        self.tokenizer = AssemblyTokenizer(base_model_name=model_name)

        self.layer_norm = nn.LayerNorm(embedding_dim)

    # ------------------------------------------------------------------
    def _mean_pool(self, hidden_states: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over real (non-padding) token positions."""
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    # ------------------------------------------------------------------
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_dict: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_dict:    If True, also return intermediate tensors

        Returns:
            (batch, embedding_dim) L2-normalised embedding, or a dict thereof
        """
        # 1. CodeBERT contextual encoding
        outputs     = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state   # (B, T, base_dim)

        # 2. RoPE-enhanced refinement attention (Q/K rotated, residual added)
        if self.use_rope:
            last_hidden = self.rope_attn(last_hidden, attention_mask)

        # 3. Mean pooling
        pooled = self._mean_pool(last_hidden, attention_mask)   # (B, base_dim)

        # 4. Projection
        embedded = self.projection(pooled)                      # (B, embedding_dim)

        # 5. Layer norm + L2 normalisation
        embedded = self.layer_norm(embedded)
        embedded = F.normalize(embedded, p=2, dim=1)

        if return_dict:
            return {
                'embedding':        embedded,
                'last_hidden_state': last_hidden,
                'attention_mask':   attention_mask,
            }
        return embedded

    def embed_trace(self,
                    trace_lines: Union[str, List[str]],
                    device: Optional[str] = None) -> torch.Tensor:
        """
        Convenience method to embed a single trace.

        Args:
            trace_lines: Trace as string or list of instructions
            device: Device to run inference on

        Returns:
            Embedding vector (embedding_dim,)
        """
        # Tokenize
        encoded = self.tokenizer.tokenize_trace(trace_lines)

        # Move to device
        if device:
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
        else:
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

        # Generate embedding
        with torch.no_grad():
            embedding = self.forward(input_ids.unsqueeze(0),
                                    attention_mask.unsqueeze(0))

        return embedding.squeeze(0)

    def embed_batch(self,
                    traces: List[Union[str, List[str]]],
                    device: Optional[str] = None,
                    batch_size: int = 8) -> torch.Tensor:
        """
        Embed multiple traces in batches.

        Args:
            traces: List of traces
            device: Device to run inference on
            batch_size: Batch size for processing

        Returns:
            Embedding matrix (num_traces, embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(traces), batch_size):
            batch_traces = traces[i:i + batch_size]

            # Tokenize all traces in batch
            batch_encoded = [self.tokenizer.tokenize_trace(t) for t in batch_traces]

            # Pad to same length
            max_len = max(enc['input_ids'].shape[0] for enc in batch_encoded)

            input_ids_list = []
            mask_list = []

            for enc in batch_encoded:
                ids = enc['input_ids']
                mask = enc['attention_mask']

                # Pad if necessary
                if len(ids) < max_len:
                    pad_len = max_len - len(ids)
                    ids = F.pad(ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                    mask = F.pad(mask, (0, pad_len), value=0)

                input_ids_list.append(ids)
                mask_list.append(mask)

            input_ids = torch.stack(input_ids_list)
            attention_mask = torch.stack(mask_list)

            # Move to device
            if device:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            # Generate embeddings
            embeddings = self.forward(input_ids, attention_mask)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def save_pretrained(self, save_path: str):
        """Save model and tokenizer."""
        print("Saving pretrained model...")
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'embedding_dim': self.embedding_dim,
                'max_seq_len': self.max_seq_len,
                'use_rope': self.use_rope
            },
            save_path,
        )

    @classmethod
    def load_pretrained(cls, load_path: str, device: Optional[str] = None):
        """Load pretrained model."""
        checkpoint = torch.load(load_path, map_location=device)

        model = cls(
            embedding_dim=checkpoint['embedding_dim'],
            max_seq_len=checkpoint['max_seq_len'],
            use_rope=checkpoint['use_rope']
        )

        model.load_state_dict(checkpoint['model_state_dict'])

        if device:
            model = model.to(device)

        return model


# =============================================================================
# DEGRADATION LOSS
# =============================================================================

class DegradationLoss(nn.Module):
    """
    Single-target MSE loss for query degradation prediction.

    Target:  deg_query = cyc_T_query_concurrent - cyc_T_query_solo  (≥ 0)

    log_scale=True applies log1p before MSE to prevent the rare high-
    interference samples (9, 18, 27 cy conflict blocks) from dominating
    every gradient update over the majority of zero-degradation pairs.
    """

    def __init__(self, log_scale: bool = True):
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch,) or (batch, 1) — raw model output
            target: (batch,) — true cycle degradation of the query

        Returns:
            Scalar MSE loss
        """
        pred   = pred.squeeze(-1)
        target = target.squeeze(-1)
        if self.log_scale:
            pred   = torch.log1p(F.relu(pred))
            target = torch.log1p(target.clamp(min=0.0))
        return F.mse_loss(pred, target)


# =============================================================================
# INTERFERENCE PREDICTOR (Single-output Bilinear Regression Head)
# =============================================================================

class InterferencePredictor(nn.Module):
    """
    Predicts how many extra cycles T_query will suffer when co-scheduled
    with T_reference, for arbitrary (query, reference) pairs.

    Because T_ref is not fixed, the model must generalise across reference
    traces it has not seen during training.  The bilinear interaction term
    (emb_q ⊙ emb_r) gives the MLP a direct signal about which cache-set
    dimensions the two traces share — the primary driver of interference —
    without requiring a fixed reference.

    Combined feature vector (dim = 3 × embedding_dim):
        [emb_q  ‖  emb_r  ‖  emb_q ⊙ emb_r]

    Output: single scalar — predicted extra cycles for T_query.

    Asymmetry is handled at the call site: to get T_ref's degradation,
    call predictor(T_ref, T_query) with the arguments swapped.
    """

    def __init__(self, embedder: TraceEmbedder):
        super().__init__()
        self.embedder = embedder
        d = embedder.embedding_dim

        self.regressor = nn.Sequential(
            nn.Linear(3 * d, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),          # → scalar deg_query
        )

    def forward(self,
                input_ids_q:  torch.Tensor,
                attn_mask_q:  torch.Tensor,
                input_ids_r:  torch.Tensor,
                attn_mask_r:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids_q / attn_mask_q : tokenised query     trace (batch, seq)
            input_ids_r / attn_mask_r : tokenised reference trace (batch, seq)

        Returns:
            (batch,) predicted cycle degradation of the query
        """
        emb_q = self.embedder(input_ids_q, attn_mask_q)   # (B, d)
        emb_r = self.embedder(input_ids_r, attn_mask_r)   # (B, d)
        combined = torch.cat([emb_q, emb_r, emb_q * emb_r], dim=1)  # (B, 3d)
        return self.regressor(combined).squeeze(-1)        # (B,)

    def predict(self,
                query_trace:     List[str],
                reference_trace: List[str],
                device: str = "cpu") -> float:
        """
        Predict the cycle degradation of query_trace when co-run with
        reference_trace.  To get the symmetric value, swap the arguments.

        Returns
        -------
        float  — predicted extra cycles (≥ 0, log-inverse transform applied)
        """
        tok = self.embedder.tokenizer

        def _encode(trace):
            enc = tok.tokenize_trace(trace)
            return (enc['input_ids'].unsqueeze(0).to(device),
                    enc['attention_mask'].unsqueeze(0).to(device))

        ids_q, mask_q = _encode(query_trace)
        ids_r, mask_r = _encode(reference_trace)

        self.eval()
        with torch.no_grad():
            raw = self(ids_q, mask_q, ids_r, mask_r)   # (1,)

        # Undo log1p: expm1(x) = e^x − 1
        return torch.expm1(raw.clamp(min=0.0)).item()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_similarity(embedding1: torch.Tensor,
                      embedding2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    if len(embedding1.shape) == 1:
        embedding1 = embedding1.unsqueeze(0)
    if len(embedding2.shape) == 1:
        embedding2 = embedding2.unsqueeze(0)

    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()


def find_similar_traces(query_embedding: torch.Tensor,
                       candidate_embeddings: torch.Tensor,
                       top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find most similar traces to a query.

    Args:
        query_embedding: Query embedding (dim,)
        candidate_embeddings: Candidate embeddings (num_candidates, dim)
        top_k: Number of results to return

    Returns:
        Tuple of (similarities, indices)
    """
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.unsqueeze(0)

    similarities = F.cosine_similarity(query_embedding, candidate_embeddings)
    top_values, top_indices = torch.topk(similarities, top_k)

    return top_values, top_indices



# =============================================================================
# INTERFERENCE ANALYSIS CLASS
# =============================================================================

class InterferenceAnalyzer:
    """
    Analyzes interference between execution traces using embeddings.
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to pretrained model (optional)
            device: Device to run on (cuda/cpu)
        """
        print("Initializing TraceEmbedder...")
        self.embedder = TraceEmbedder(
            model_name="microsoft/codebert-base",
            embedding_dim=768,
            use_rope=True
        )

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embedder = self.embedder.to(self.device)
        self.embedder.eval()

        # Load pretrained weights if provided
        if model_path:
            print(f"Loading pretrained model from {model_path}...")
            self.embedder = TraceEmbedder.load_pretrained(model_path, device=self.device)

        print(f"✓ Analyzer ready on {self.device}")

    def embed_trace(self, trace: List[str], name: str = "Trace") -> torch.Tensor:
        """
        Embed a single trace.

        Args:
            trace: List of assembly instructions
            name: Trace name for logging

        Returns:
            Embedding tensor
        """
        print(f"  Embedding {name} ({len(trace)} instructions)...")
        with torch.no_grad():
            embedding = self.embedder.embed_trace(trace, device=self.device)
        return embedding

    def compute_distance(self, emb1: torch.Tensor, emb2: torch.Tensor,
                        metric: str = "cosine") -> float:
        """
        Compute distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding
            metric: Distance metric (cosine, euclidean, manhattan)

        Returns:
            Distance score
        """
        if metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            return 1 - similarity.item()

        elif metric == "euclidean":
            return torch.dist(emb1, emb2, p=2).item()

        elif metric == "manhattan":
            return torch.dist(emb1, emb2, p=1).item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        return compute_similarity(emb1, emb2)

    def analyze_pair(self, trace1: List[str], trace2: List[str],
                    name1: str = "Trace 1", name2: str = "Trace 2") -> Dict:
        """
        Analyze interference between two traces.

        Args:
            trace1: First trace
            trace2: Second trace
            name1: Name for first trace
            name2: Name for second trace

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Interference: {name1} vs {name2}")
        print(f"{'='*60}")

        # Generate embeddings
        emb1 = self.embed_trace(trace1, name1)
        emb2 = self.embed_trace(trace2, name2)

        # Compute metrics
        cosine_sim = self.compute_similarity(emb1, emb2)
        cosine_dist = self.compute_distance(emb1, emb2, "cosine")
        euclidean_dist = self.compute_distance(emb1, emb2, "euclidean")
        manhattan_dist = self.compute_distance(emb1, emb2, "manhattan")

        # Interpret interference level
        if cosine_sim > 0.85:
            interference_level = "HIGH"
            interference_desc = "Likely to cause cache conflicts and resource contention"
        elif cosine_sim > 0.65:
            interference_level = "MEDIUM"
            interference_desc = "May cause some interference depending on timing"
        else:
            interference_level = "LOW"
            interference_desc = "Unlikely to cause significant interference"

        results = {
            'trace1_name': name1,
            'trace2_name': name2,
            'cosine_similarity': cosine_sim,
            'cosine_distance': cosine_dist,
            'euclidean_distance': euclidean_dist,
            'manhattan_distance': manhattan_dist,
            'interference_level': interference_level,
            'interference_description': interference_desc,
            'embedding1_norm': torch.norm(emb1).item(),
            'embedding2_norm': torch.norm(emb2).item()
        }

        # Print results
        self._print_results(results)

        return results

    def _print_results(self, results: Dict):
        """Pretty print analysis results."""
        print(f"\n📊 Results:")
        print(f"  Cosine Similarity:  {results['cosine_similarity']:.4f}")
        print(f"  Cosine Distance:    {results['cosine_distance']:.4f}")
        print(f"  Euclidean Distance: {results['euclidean_distance']:.4f}")
        print(f"  Manhattan Distance: {results['manhattan_distance']:.4f}")
        print(f"\n⚠️  Interference Level: {results['interference_level']}")
        print(f"  {results['interference_description']}")
        print(f"{'='*60}\n")

    def analyze_multiple(self, traces: Dict[str, List[str]]) -> Dict:
        """
        Analyze all pairs of traces.

        Args:
            traces: Dictionary of {name: trace_instructions}

        Returns:
            Dictionary with all pairwise comparisons
        """
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS: {len(traces)} Traces")
        print(f"{'='*60}")

        # Embed all traces
        embeddings = {}
        for name, trace in traces.items():
            embeddings[name] = self.embed_trace(trace, name)

        # Compute all pairwise similarities
        results = {}
        trace_names = list(traces.keys())

        print(f"\n📈 Similarity Matrix:")
        print(f"{'':15}", end="")
        for name in trace_names:
            print(f"{name[:12]:12}", end=" ")
        print()

        for i, name1 in enumerate(trace_names):
            print(f"{name1[:15]:15}", end="")
            results[name1] = {}
            for j, name2 in enumerate(trace_names):
                if i == j:
                    print(f"{'1.0000':12}", end=" ")
                    results[name1][name2] = 1.0
                else:
                    sim = self.compute_similarity(embeddings[name1], embeddings[name2])
                    results[name1][name2] = sim
                    print(f"{sim:.4f}:12", end=" ")
            print()

        return {
            'embeddings': embeddings,
            'pairwise_similarities': results
        }


# =============================================================================
# VISUALIZATION HELPER
# =============================================================================

def print_interference_report(results: List[Dict]):
    """
    Print a summary report of all interference analyses.

    Args:
        results: List of analysis result dictionaries
    """
    print(f"\n{'='*60}")
    print(f"📋 INTERFERENCE ANALYSIS SUMMARY")
    print(f"{'='*60}\n")

    # Sort by similarity (highest first)
    sorted_results = sorted(results, key=lambda x: x['cosine_similarity'], reverse=True)

    print(f"{'Trace Pair':40} {'Similarity':12} {'Level':10}")
    print(f"{'-'*60}")

    for r in sorted_results:
        pair_name = f"{r['trace1_name']} vs {r['trace2_name']}"
        print(f"{pair_name[:40]:40} {r['cosine_similarity']:12.4f} {r['interference_level']:10}")

    print(f"\n{'='*60}")

    # Statistics
    similarities = [r['cosine_similarity'] for r in results]
    print(f"\n📊 Statistics:")
    print(f"  Average Similarity: {sum(similarities)/len(similarities):.4f}")
    print(f"  Max Similarity:     {max(similarities):.4f}")
    print(f"  Min Similarity:     {min(similarities):.4f}")
    print(f"  High Interference Pairs: {sum(1 for r in results if r['interference_level'] == 'HIGH')}")
    print(f"  Medium Interference Pairs: {sum(1 for r in results if r['interference_level'] == 'MEDIUM')}")
    print(f"  Low Interference Pairs: {sum(1 for r in results if r['interference_level'] == 'LOW')}")


# generate_training_data.py
import random

def generate_trace_pair_org(positive=True):
    """Generate trace pairs with known interference labels."""

    if positive:
        # Same cache sets (will interfere)
        base_addr = random.choice([0x2000, 0x3000, 0x4000])
        trace1 = [f"LDR R0, [0x{base_addr + i*4:X}]" for i in range(10)]
        trace2 = [f"LDR R1, [0x{base_addr + i*4:X}]" for i in range(10)]
        label = 1  # Positive (will interfere)
    else:
        # Different cache sets (won't interfere)
        addr1 = random.choice([0x2000, 0x3000])
        addr2 = random.choice([0x5000, 0x6000])
        trace1 = [f"LDR R0, [0x{addr1 + i*4:X}]" for i in range(10)]
        trace2 = [f"LDR R1, [0x{addr2 + i*4:X}]" for i in range(10)]
        label = 0  # Negative (won't interfere)

    return trace1, trace2, label

# train_interference_model.py

import random
from torch.utils.data import DataLoader
import torch

def _directed_labels_from_pair(trace1: List[str],
                                trace2: List[str]) -> Tuple[float, float]:
    """
    Run the simulator and return the directed degradation for each trace.

    Returns
    -------
    deg1 : float  — extra cycles T1 suffers when co-run with T2  (≥ 0)
    deg2 : float  — extra cycles T2 suffers when co-run with T1  (≥ 0)
    """
    from generate_trace_pair import run_solo, run_concurrent
    solo0        = run_solo(trace1)
    solo1        = run_solo(trace2)
    conc0, conc1 = run_concurrent(trace1, trace2)
    return float(conc0 - solo0), float(conc1 - solo1)


def train_model(num_epochs: int = 100, batch_size: int = 8):
    """
    Train InterferencePredictor to predict query cycle degradation for
    arbitrary (query, reference) pairs.

    Label per sample
    ----------------
    target = cyc_T_query_concurrent - cyc_T_query_solo   (scalar, ≥ 0)

    Both orderings of each generated pair are added to the batch:
        (T1 as query, T2 as reference) → target = deg1
        (T2 as query, T1 as reference) → target = deg2

    This teaches the model the asymmetry of interference (deg1 ≠ deg2 in
    general) and doubles training signal without extra simulator calls.
    A diverse mix of low / medium / high interference pairs is generated
    so the model sees a wide range of reference behaviours.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embedder   = TraceEmbedder().to(device)
    predictor  = InterferencePredictor(embedder).to(device)
    optimizer  = torch.optim.AdamW(predictor.parameters(), lr=2e-5)
    loss_fn    = DegradationLoss(log_scale=True)

    tok = embedder.tokenizer

    def _encode_batch(traces: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a list of traces and stack into (B, seq_len) tensors."""
        encoded = [tok.tokenize_trace(t) for t in traces]
        max_len = max(e['input_ids'].shape[0] for e in encoded)
        ids, masks = [], []
        for e in encoded:
            pad = max_len - e['input_ids'].shape[0]
            ids.append(F.pad(e['input_ids'],        (0, pad), value=tok.pad_token_id))
            masks.append(F.pad(e['attention_mask'], (0, pad), value=0))
        return (torch.stack(ids).to(device),
                torch.stack(masks).to(device))

    for epoch in range(num_epochs):
        predictor.train()
        total_loss = 0.0

        # ── build batch (forward + reversed = 2 × batch_size samples) ────────
        queries, references, targets = [], [], []

        for _ in range(batch_size):
            t1, t2 = generate_trace_pair()
            deg1, deg2 = _directed_labels_from_pair(t1, t2)

            # Forward:  query=T1, reference=T2  →  target = deg1
            queries.append(t1);  references.append(t2);  targets.append(deg1)

            # Reversed: query=T2, reference=T1  →  target = deg2
            queries.append(t2);  references.append(t1);  targets.append(deg2)

        # ── encode ────────────────────────────────────────────────────────────
        ids_q,  mask_q = _encode_batch(queries)
        ids_r,  mask_r = _encode_batch(references)
        target_tensor  = torch.tensor(targets, dtype=torch.float32).to(device)
        # shapes: ids (2B, seq), target (2B,)

        # ── forward + loss ────────────────────────────────────────────────────
        pred = predictor(ids_q, mask_q, ids_r, mask_r)   # (2B,)
        loss = loss_fn(pred, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{num_epochs}  loss={total_loss:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({
        'embedder_state':  embedder.state_dict(),
        'predictor_state': predictor.state_dict(),
        'embedding_dim':   embedder.embedding_dim,
        'max_seq_len':     embedder.max_seq_len,
        'use_rope':        embedder.use_rope,
    }, "interference_predictor.pt")
    print("✓ Model trained and saved to interference_predictor.pt")


if __name__ == "__main__":
    train_model()

