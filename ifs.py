"""
Execution Trace Embedding Module
=====================================
A Transformer-based embedding system for simple assembly execution traces.

Usage:
    embedder = TraceEmbedder()
    embedding = embedder.embed_trace(trace_lines)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast
import math
import random



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
# PROBE REGRESSION HEAD (metric-learning scheme)
# =============================================================================

class ProbeRegressionHead(nn.Module):
    """
    Maps a single query embedding to a predicted degradation profile: one
    cycle-degradation scalar per probe in the fixed probe pool.

    Training objective
    ------------------
    A fixed pool of N_PROBES diverse reference traces is generated once
    (generate_probe_pool) and held constant throughout training.  For each
    query trace q, the target is the vector:

        y[i] = run_concurrent(q, probe[i])[0] - run_solo(q)   (≥ 0)

    The model is trained to minimise MSE between pred and y over all probes
    simultaneously.  This directly shapes the embedding space:

        Two queries q1, q2 with similar degradation profiles (y1 ≈ y2)
        will be pulled together in embedding space, so cosine similarity
        between embeddings becomes a *principled* proxy for interference
        susceptibility similarity — unlike the original scheme where the
        reference varied freely across training samples.

    Architecture
    ------------
    embedding (d) → Linear(d, 256) → GELU → Dropout(0.1)
                  → Linear(256, 128) → GELU → Dropout(0.1)
                  → Linear(128, n_probes)   → predicted profile (n_probes,)

    The head is intentionally lightweight: the heavy lifting (capturing
    cache-set access patterns) is done by TraceEmbedder; the head only
    needs to learn the mapping from embedding space to degradation space.
    """

    def __init__(self, embedder: TraceEmbedder, n_probes: int):
        super().__init__()
        self.embedder  = embedder
        self.n_probes  = n_probes
        d = embedder.embedding_dim

        self.head = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_probes),   # → predicted degradation profile
        )

    def forward(self,
                input_ids:    torch.Tensor,
                attn_mask:    torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids / attn_mask : tokenised query trace (batch, seq)

        Returns:
            (batch, n_probes) predicted cycle degradation against each probe
        """
        emb = self.embedder(input_ids, attn_mask)   # (B, d)
        return self.head(emb)                        # (B, n_probes)

    def predict_profile(self,
                        query_trace: List[str],
                        device: str = "cpu") -> torch.Tensor:
        """
        Predict the degradation profile of a single query trace.

        Returns
        -------
        (n_probes,) tensor of predicted extra cycles against each probe (≥ 0).
        """
        tok = self.embedder.tokenizer
        enc = tok.tokenize_trace(query_trace)
        ids  = enc['input_ids'].unsqueeze(0).to(device)
        mask = enc['attention_mask'].unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            raw = self(ids, mask)   # (1, n_probes)
        return F.relu(raw.squeeze(0))   # clip negatives; degradation ≥ 0


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
    Analyzes interference susceptibility between execution traces.

    With the probe-based metric-learning scheme, cosine similarity between
    two query embeddings has a precise meaning: traces that are close in
    embedding space have similar degradation profiles across the fixed probe
    pool, i.e. they suffer from similar levels of cache interference against
    the same canonical set of reference workloads.

    The analyzer exposes two complementary views:

    1. Embedding similarity (analyze_pair): fast cosine similarity between
       the L2-normalised embeddings — the primary signal for susceptibility
       comparison.

    2. Profile comparison (analyze_profiles): run both traces through the
       ProbeRegressionHead to obtain their predicted degradation vectors,
       then compare those vectors directly.  This is slower but gives an
       interpretable, per-probe breakdown of where the interference comes from.
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to checkpoint saved by train_model() (optional).
                        If provided, also loads the probe pool and head weights.
            device: Device to run on (cuda/cpu).
        """
        print("Initializing TraceEmbedder...")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.embedder  = None
        self.head      = None
        self.probes    = None

        if model_path:
            print(f"Loading pretrained model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)

            self.embedder = TraceEmbedder(
                embedding_dim=checkpoint['embedding_dim'],
                max_seq_len=checkpoint['max_seq_len'],
                use_rope=checkpoint['use_rope'],
            )
            self.embedder.load_state_dict(checkpoint['embedder_state'])

            n_probes = checkpoint['n_probes']
            self.head = ProbeRegressionHead(self.embedder, n_probes=n_probes)
            self.head.head.load_state_dict(checkpoint['head_state'])
            self.head = self.head.to(self.device)

            self.probes = checkpoint.get('probes', None)
            print(f"✓ Loaded model with {n_probes} probes")
        else:
            self.embedder = TraceEmbedder(
                model_name="microsoft/codebert-base",
                embedding_dim=768,
                use_rope=True,
            )

        self.embedder = self.embedder.to(self.device)
        self.embedder.eval()
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

    def analyze_profiles(self,
                         trace1: List[str],
                         trace2: List[str],
                         name1: str = "Trace 1",
                         name2: str = "Trace 2") -> Dict:
        """
        Compare two traces via their predicted degradation profiles.

        Requires a trained ProbeRegressionHead (model_path must have been
        provided to __init__).  Returns the cosine similarity between the
        two predicted profile vectors in addition to per-probe breakdowns.

        This is the interpretable complement to analyze_pair: instead of
        comparing embeddings, it compares the predicted degradation vectors
        directly, showing *which probes* drive the similarity or difference.

        Args:
            trace1, trace2 : assembly instruction lists
            name1, name2   : display names

        Returns:
            Dict with profile vectors, per-probe breakdown, and similarity.
        """
        if self.head is None:
            raise RuntimeError(
                "analyze_profiles requires a trained model. "
                "Pass model_path= when constructing InterferenceAnalyzer."
            )

        prof1 = self.head.predict_profile(trace1, device=self.device)  # (n_probes,)
        prof2 = self.head.predict_profile(trace2, device=self.device)

        sim = F.cosine_similarity(prof1.unsqueeze(0), prof2.unsqueeze(0)).item()
        diff = (prof1 - prof2).abs()

        results = {
            'trace1_name':       name1,
            'trace2_name':       name2,
            'profile1':          prof1.cpu().tolist(),
            'profile2':          prof2.cpu().tolist(),
            'profile_cosine_sim': sim,
            'mean_deg1':         prof1.mean().item(),
            'mean_deg2':         prof2.mean().item(),
            'max_diff_probe':    diff.argmax().item(),
            'max_diff_value':    diff.max().item(),
        }

        print(f"\n{'='*60}")
        print(f"Profile comparison: {name1} vs {name2}")
        print(f"  Profile cosine similarity : {sim:.4f}")
        print(f"  Mean predicted degradation: {name1}={results['mean_deg1']:.2f} cy  "
              f"{name2}={results['mean_deg2']:.2f} cy")
        print(f"  Largest per-probe diff    : probe {results['max_diff_probe']} "
              f"({results['max_diff_value']:.2f} cy)")
        print(f"{'='*60}\n")
        return results

    def analyze_pair(self, trace1: List[str], trace2: List[str],
                    name1: str = "Trace 1", name2: str = "Trace 2") -> Dict:
        """
        Compare the interference susceptibility of two traces via embedding
        cosine similarity.

        With the probe-based training scheme, two traces that are close in
        embedding space have provably similar degradation profiles across the
        fixed probe pool — so this similarity has a precise, principled
        meaning rather than being a heuristic proxy.

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





def train_model(
    num_epochs: int   = 100,
    batch_size: int   = 4,
    n_probes:   int   = 32,
    probe_seed: int   = 42,
    trace_len:  int   = 16,
    nop_prob:   float = 0.25,
):
    """
    Train ProbeRegressionHead with a fixed probe pool (metric-learning scheme).

    Training objective
    ------------------
    A pool of *n_probes* diverse reference traces is generated once with a
    fixed seed and held constant for the entire training run.  For each query
    trace q in a batch, the simulator computes the degradation profile:

        y[i] = run_concurrent(q, probe[i])[0] - run_solo(q)   for i in probes

    The model predicts this full profile from the query embedding alone, and
    MSE loss is computed across all probe dimensions simultaneously.

    Why this makes cosine similarity meaningful
    -------------------------------------------
    Two queries q1, q2 that suffer similarly across all probes will have
    similar target vectors y1 ≈ y2.  Minimising MSE forces their embeddings
    to be similar, so after training, cosine similarity between two query
    embeddings directly reflects how similar their interference susceptibility
    profiles are — regardless of which specific co-runner they face at
    inference time.

    This is in contrast to the original scheme where the reference varied
    freely: there, the embedding had no stable geometric meaning because
    "similar embedding" did not imply "similar degradation against the same
    reference."

    Probe diversity
    ---------------
    Probes are generated fully uniformly (bias_prob=0.0) so they cover the
    256 L2 sets evenly.  With n_probes=32 probes × ~12 LDRs each, roughly
    384 distinct cache-set accesses are represented in the probe pool, giving
    good coverage of the interference space at 16 KB / 256 sets.

    Query generation
    ----------------
    Query traces are generated with bias_prob=0.5 so that roughly half of
    them will produce non-zero degradation against at least one probe,
    preventing the model from collapsing to always predicting zero.
    """
    from generate_trace_pair import (
        generate_probe_pool,
        compute_degradation_profile,
        generate_trace_pair,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── build fixed probe pool (generated once, held constant) ────────────────
    print(f"Generating probe pool ({n_probes} probes, seed={probe_seed})...")
    probes = generate_probe_pool(n_probes=n_probes, trace_len=trace_len,
                                 nop_prob=nop_prob, seed=probe_seed)
    print(f"✓ Probe pool ready ({n_probes} traces × {trace_len} instructions)")

    # ── model, optimiser ──────────────────────────────────────────────────────
    embedder = TraceEmbedder().to(device)
    model    = ProbeRegressionHead(embedder, n_probes=n_probes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    tok = embedder.tokenizer

    def _encode_batch(traces: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a list of query traces and stack into (B, seq_len) tensors."""
        encoded = [tok.tokenize_trace(t) for t in traces]
        max_len = max(e['input_ids'].shape[0] for e in encoded)
        ids, masks = [], []
        for e in encoded:
            pad = max_len - e['input_ids'].shape[0]
            ids.append(F.pad(e['input_ids'],        (0, pad), value=tok.pad_token_id))
            masks.append(F.pad(e['attention_mask'], (0, pad), value=0))
        return (torch.stack(ids).to(device),
                torch.stack(masks).to(device))

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # ── build batch ───────────────────────────────────────────────────────
        # Generate query traces with bias_prob=0.5 so that a good fraction
        # have non-zero interference against at least some probes.
        queries  = []
        targets  = []   # list of (n_probes,) degradation profile vectors

        for _ in range(batch_size):
            # generate_trace_pair returns (t1, t2); we use t1 as the query.
            # t2 is discarded — the probe pool is the fixed reference set.
            t1, _ = generate_trace_pair(trace_len=trace_len,
                                        nop_prob=nop_prob,
                                        bias_prob=0.5)
            profile = compute_degradation_profile(t1, probes)
            queries.append(t1)
            targets.append(profile)

        # ── encode queries ────────────────────────────────────────────────────
        ids_q, mask_q = _encode_batch(queries)

        # target shape: (B, n_probes)
        target_tensor = torch.tensor(targets, dtype=torch.float32).to(device)

        # ── forward + MSE loss over all probe dimensions ──────────────────────
        pred = model(ids_q, mask_q)                           # (B, n_probes)
        loss = F.mse_loss(pred, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            nonzero = (target_tensor > 0).float().mean().item()
            print(f"Epoch {epoch+1:4d}/{num_epochs}  "
                  f"loss={total_loss:.4f}  "
                  f"pred_mean={pred.mean().item():+.3f}  "
                  f"target_mean={target_tensor.mean().item():.3f}  "
                  f"nonzero_frac={nonzero:.2f}")

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({
        'embedder_state': embedder.state_dict(),
        'head_state':     model.head.state_dict(),
        'embedding_dim':  embedder.embedding_dim,
        'max_seq_len':    embedder.max_seq_len,
        'use_rope':       embedder.use_rope,
        'n_probes':       n_probes,
        'probe_seed':     probe_seed,
        'trace_len':      trace_len,
        'nop_prob':       nop_prob,
        # Store the probe pool itself so inference can reproduce profiles
        'probes':         probes,
    }, "interference_predictor.pt")
    print("✓ Model trained and saved to interference_predictor.pt")




if __name__ == "__main__":
    train_model()

