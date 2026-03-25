"""
models.py — Character-Level Name Generation Models (From Scratch)
=================================================================

This module implements three recurrent neural network architectures for
character-level name generation.  All recurrent cells are coded from
scratch (no nn.RNN / nn.LSTM wrappers):

  1. Vanilla RNN          — multi-layer Elman-cell language model
  2. Bidirectional LSTM   — BiLSTM encoder + unidirectional LSTM decoder
  3. RNN with Attention   — RNN encoder-decoder with Bahdanau attention

Author : Auto-generated for NLU Problem 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================================================
#  1.  VANILLA RNN  (hand-coded Elman cell)
# ==========================================================================

class VanillaRNNCell(nn.Module):
    """
    Single Elman RNN cell implemented from scratch.

    Computes:
        h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)

    Parameters
    ----------
    input_size  : int  — dimensionality of the input vector at each timestep
    hidden_size : int  — dimensionality of the hidden state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Learnable weight matrices  (Xavier-style initialisation)
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_size) / math.sqrt(input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) / math.sqrt(hidden_size))
        self.b    = nn.Parameter(torch.zeros(hidden_size))        # bias term

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one timestep.

        Parameters
        ----------
        x : Tensor of shape (batch, input_size)   — current input
        h : Tensor of shape (batch, hidden_size)   — previous hidden state

        Returns
        -------
        h_new : Tensor of shape (batch, hidden_size) — new hidden state
        """
        return torch.tanh(x @ self.W_ih + h @ self.W_hh + self.b)


class VanillaRNN(nn.Module):
    """
    Multi-layer Vanilla RNN language model for character-level generation.

    Architecture:
        Embedding → Stacked RNN Cells (num_layers) → Dropout → Linear → Logits

    This is a *language model* (not encoder-decoder): it predicts the next
    character given all previous characters, so training and generation
    both proceed left-to-right in the same way.

    Parameters
    ----------
    vocab_size  : int   — size of the character vocabulary
    embed_size  : int   — character embedding dimension
    hidden_size : int   — RNN hidden state dimension
    num_layers  : int   — number of stacked RNN layers
    dropout     : float — dropout probability applied between layers
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Character embedding table:  index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Build one RNN cell per layer (input size differs for layer 0 vs deeper)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            inp_size = embed_size if i == 0 else hidden_size
            self.cells.append(VanillaRNNCell(inp_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

        # Output projection:  hidden state → logits over vocabulary
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Forward pass over a full sequence (teacher-forcing mode).

        Parameters
        ----------
        x      : LongTensor of shape (batch, seq_len) — input token indices
        hidden : optional list of hidden states for each layer

        Returns
        -------
        logits : Tensor (batch, seq_len, vocab_size) — unnormalised scores
        hidden : list[Tensor] — final hidden state for each layer
        """
        B, T = x.shape
        device = x.device

        # Initialise hidden states to zero if not provided
        if hidden is None:
            hidden = [torch.zeros(B, self.hidden_size, device=device)
                      for _ in range(self.num_layers)]

        # Embed the input tokens:  (B, T) → (B, T, embed_size)
        emb = self.dropout(self.embedding(x))

        outputs = []
        for t in range(T):
            inp = emb[:, t, :]                       # (B, E)  — embedding at time t

            new_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                h = cell(inp, hidden[layer_idx])      # apply RNN cell
                inp = self.dropout(h)                  # feed to next layer (with dropout)
                new_hidden.append(h)
            hidden = new_hidden
            outputs.append(hidden[-1])                 # record top-layer hidden state

        # Stack timestep outputs → (B, T, H) then project to vocab logits
        out = torch.stack(outputs, dim=1)
        logits = self.fc_out(out)                      # (B, T, vocab_size)
        return logits, hidden

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================================================
#  2.  BIDIRECTIONAL LSTM  (encoder-decoder)
# ==========================================================================

class LSTMCell(nn.Module):
    """
    Single LSTM cell implemented from scratch.

    Computes the four gates (input, forget, cell, output) in one fused
    matrix multiply for efficiency:

        [i, f, g, o] = x · W_x + h · W_h + bias
        i = σ(i),  f = σ(f),  g = tanh(g),  o = σ(o)
        c_new = f * c + i * g
        h_new = o * tanh(c_new)

    Parameters
    ----------
    input_size  : int — input vector dimensionality
    hidden_size : int — hidden/cell state dimensionality
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined weight matrices for all four gates  (4H columns)
        self.W_x  = nn.Parameter(torch.randn(input_size, 4 * hidden_size) / math.sqrt(input_size))
        self.W_h  = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) / math.sqrt(hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        # Initialise forget-gate bias to 1.0 so the cell retains information
        # early in training  (a well-known best practice)
        nn.init.constant_(self.bias[hidden_size:2*hidden_size], 1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        Forward pass for one timestep.

        Parameters
        ----------
        x : Tensor (batch, input_size)   — current input
        h : Tensor (batch, hidden_size)  — previous hidden state
        c : Tensor (batch, hidden_size)  — previous cell state

        Returns
        -------
        h_new, c_new : Tensors of shape (batch, hidden_size)
        """
        gates = x @ self.W_x + h @ self.W_h + self.bias     # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=-1)                  # split into 4 gates

        i = torch.sigmoid(i)       # input gate
        f = torch.sigmoid(f)       # forget gate
        g = torch.tanh(g)          # candidate cell content
        o = torch.sigmoid(o)       # output gate

        c_new = f * c + i * g      # new cell state
        h_new = o * torch.tanh(c_new)  # new hidden state
        return h_new, c_new


class BLSTMGenerator(nn.Module):
    """
    Bidirectional LSTM encoder + unidirectional LSTM decoder for name generation.

    Architecture:
        ENCODER:  Embedding → 2-layer BiLSTM (hand-coded cells)
                  Produces per-timestep outputs and final (h, c) states.
        BRIDGE:   Linear projections  (2*H → H)  for h and c.
        DECODER:  2-layer uni-directional LSTM (hand-coded cells)
                  Takes encoder context as initial state.

    Training uses teacher forcing.  At generation time the decoder is
    initialised with zero states (since there is no encoder input).

    Parameters
    ----------
    vocab_size  : int
    embed_size  : int
    hidden_size : int
    num_layers  : int
    dropout     : float
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Shared embedding for both encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # ── Encoder: bi-directional LSTM cells ───────────────────────────
        self.enc_fwd_cells = nn.ModuleList()   # forward-direction cells
        self.enc_bwd_cells = nn.ModuleList()   # backward-direction cells
        for i in range(num_layers):
            # Layer 0 takes embedding; deeper layers take concatenated fwd+bwd output
            inp_size = embed_size if i == 0 else hidden_size * 2
            self.enc_fwd_cells.append(LSTMCell(inp_size, hidden_size))
            self.enc_bwd_cells.append(LSTMCell(inp_size, hidden_size))

        # ── Decoder: uni-directional LSTM cells ──────────────────────────
        self.dec_cells = nn.ModuleList()
        for i in range(num_layers):
            inp_size = embed_size if i == 0 else hidden_size
            self.dec_cells.append(LSTMCell(inp_size, hidden_size))

        # ── Bridge: project encoder final states to decoder dimensions ───
        # Encoder produces 2*hidden_size (fwd + bwd); decoder needs hidden_size
        self.h_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.c_proj = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden_size, vocab_size)

    def _run_encoder(self, x: torch.Tensor):
        """
        Run the bidirectional LSTM encoder over the input sequence.

        Parameters
        ----------
        x : LongTensor (batch, seq_len)

        Returns
        -------
        dec_h0 : Tensor (batch, hidden_size) — initial decoder hidden state
        dec_c0 : Tensor (batch, hidden_size) — initial decoder cell state
        """
        B, T = x.shape
        device = x.device
        emb = self.dropout(self.embedding(x))   # (B, T, E)

        layer_input = emb
        final_h_fwd, final_c_fwd = [], []
        final_h_bwd, final_c_bwd = [], []

        for layer in range(self.num_layers):
            H = self.hidden_size

            # ── Forward pass through the sequence ────────────────────────
            h_f = torch.zeros(B, H, device=device)
            c_f = torch.zeros(B, H, device=device)
            fwd_outputs = []
            for t in range(T):
                h_f, c_f = self.enc_fwd_cells[layer](layer_input[:, t, :], h_f, c_f)
                fwd_outputs.append(h_f)
            final_h_fwd.append(h_f)
            final_c_fwd.append(c_f)

            # ── Backward pass through the sequence ───────────────────────
            h_b = torch.zeros(B, H, device=device)
            c_b = torch.zeros(B, H, device=device)
            bwd_outputs = []
            for t in range(T - 1, -1, -1):
                h_b, c_b = self.enc_bwd_cells[layer](layer_input[:, t, :], h_b, c_b)
                bwd_outputs.insert(0, h_b)
            final_h_bwd.append(h_b)
            final_c_bwd.append(c_b)

            # ── Concatenate forward + backward for next layer ────────────
            fwd_stack = torch.stack(fwd_outputs, dim=1)   # (B, T, H)
            bwd_stack = torch.stack(bwd_outputs, dim=1)   # (B, T, H)
            layer_input = self.dropout(
                torch.cat([fwd_stack, bwd_stack], dim=-1)  # (B, T, 2H)
            )

        # ── Bridge: project last layer's final states to decoder size ────
        h_enc = torch.cat([final_h_fwd[-1], final_h_bwd[-1]], dim=-1)   # (B, 2H)
        c_enc = torch.cat([final_c_fwd[-1], final_c_bwd[-1]], dim=-1)   # (B, 2H)

        dec_h0 = torch.tanh(self.h_proj(h_enc))   # (B, H)
        dec_c0 = torch.tanh(self.c_proj(c_enc))    # (B, H)
        return dec_h0, dec_c0

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Teacher-forcing forward pass.

        The encoder reads the full name bidirectionally.  The decoder then
        predicts next characters using ground-truth previous characters.

        Parameters
        ----------
        x : LongTensor (batch, seq_len) — full name [SOS, c1 … cn, EOS]

        Returns
        -------
        logits : Tensor (batch, seq_len, vocab_size)
        None   : (hidden not used externally for encoder-decoder models)
        """
        B, T = x.shape

        # Step 1: Encode the full sequence bidirectionally
        dec_h, dec_c = self._run_encoder(x)

        # Step 2: Decode with teacher forcing
        emb = self.dropout(self.embedding(x))
        # Initialise all decoder layers with the same projected encoder state
        hidden_states = [[dec_h, dec_c] for _ in range(self.num_layers)]

        outputs = []
        for t in range(T):
            inp = emb[:, t, :]                  # ground-truth embedding at time t
            new_hidden = []
            for layer, cell in enumerate(self.dec_cells):
                h, c = cell(inp, hidden_states[layer][0], hidden_states[layer][1])
                inp = self.dropout(h)
                new_hidden.append([h, c])
            hidden_states = new_hidden
            outputs.append(hidden_states[-1][0])  # top-layer hidden state

        out = torch.stack(outputs, dim=1)   # (B, T, H)
        logits = self.fc_out(out)           # (B, T, V)
        return logits, None

    def generate(self, sos_idx: int, eos_idx: int, vocab, device,
                 max_len: int = 30, temperature: float = 1.0) -> str:
        """
        Autoregressively generate a name (no encoder input — zero-state init).

        Parameters
        ----------
        sos_idx     : int        — index of <SOS> token
        eos_idx     : int        — index of <EOS> token
        vocab       : CharVocab  — vocabulary for decoding
        device      : torch.device
        max_len     : int        — maximum characters to generate
        temperature : float      — sampling temperature (higher = more random)

        Returns
        -------
        str — generated name string
        """
        self.eval()
        with torch.no_grad():
            # Start decoder from zero states (no encoder context)
            h = torch.zeros(1, self.hidden_size, device=device)
            c = torch.zeros(1, self.hidden_size, device=device)
            hidden_states = [[h, c] for _ in range(self.num_layers)]

            token_idx = sos_idx
            result = []
            for _ in range(max_len):
                # Embed current token
                tok_tensor = torch.tensor([token_idx], device=device)
                emb = self.embedding(tok_tensor)          # (1, E)

                # Pass through decoder layers
                inp = emb
                new_hidden = []
                for layer, cell in enumerate(self.dec_cells):
                    h_l, c_l = cell(inp, hidden_states[layer][0], hidden_states[layer][1])
                    inp = h_l
                    new_hidden.append([h_l, c_l])
                hidden_states = new_hidden

                # Sample next token with temperature scaling
                logits = self.fc_out(hidden_states[-1][0]) / temperature
                probs  = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

                if next_tok == eos_idx:
                    break
                result.append(next_tok)
                token_idx = next_tok

        return vocab.decode(result)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================================================
#  3.  RNN WITH BAHDANAU (ADDITIVE) ATTENTION
# ==========================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes:
        energy_t = V^T · tanh(W_h · h_enc + W_s · s_dec)
        α        = softmax(energy)
        context  = Σ α_t · h_enc_t

    This allows the decoder to "look back" at all encoder timesteps
    and focus on the most relevant parts when predicting each character.

    Parameters
    ----------
    enc_hidden : int — encoder hidden state dimension
    dec_hidden : int — decoder hidden state dimension
    attn_size  : int — internal attention projection dimension
    """

    def __init__(self, enc_hidden: int, dec_hidden: int, attn_size: int = 64):
        super().__init__()
        self.W_h = nn.Linear(enc_hidden, attn_size, bias=False)  # projects encoder states
        self.W_s = nn.Linear(dec_hidden, attn_size, bias=False)  # projects decoder state
        self.V   = nn.Linear(attn_size, 1, bias=False)           # score projection

    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        Compute attention context vector and weights.

        Parameters
        ----------
        encoder_outputs : Tensor (B, T_enc, enc_hidden) — all encoder hidden states
        decoder_hidden  : Tensor (B, dec_hidden)         — current decoder hidden state

        Returns
        -------
        context : Tensor (B, enc_hidden)   — weighted sum of encoder states
        weights : Tensor (B, T_enc)        — attention weight distribution
        """
        # Compute attention energies
        # W_h(encoder) → (B, T, A)   +   W_s(decoder).unsqueeze(1) → (B, 1, A)
        energy  = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden).unsqueeze(1))
        scores  = self.V(energy).squeeze(-1)               # (B, T_enc)
        weights = F.softmax(scores, dim=-1)                 # normalised attention weights

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_H)
        return context, weights


class AttentionRNN(nn.Module):
    """
    RNN language model with Bahdanau self-attention for name generation.

    Unlike an encoder-decoder, this model processes the sequence left-to-right
    just like VanillaRNN.  At each timestep the RNN hidden state attends
    back over ALL previous hidden states (self-attention), producing a
    context vector that is combined with the current embedding to predict
    the next character.

    Architecture:
        Embedding → RNN layers → Self-Attention over past hidden states
                  → Concat [hidden ; context] → Linear → Logits

    Because this is a language model (not encoder-decoder), training and
    generation are perfectly aligned — no exposure bias.

    Parameters
    ----------
    vocab_size  : int
    embed_size  : int
    hidden_size : int
    num_layers  : int
    dropout     : float
    attn_size   : int — attention projection dimension
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.2, attn_size: int = 64):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # ── RNN cells (language model style — left-to-right) ─────────────
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            inp_size = embed_size if i == 0 else hidden_size
            self.cells.append(VanillaRNNCell(inp_size, hidden_size))

        # ── Self-attention over past hidden states ───────────────────────
        self.attention = BahdanauAttention(hidden_size, hidden_size, attn_size)

        self.dropout = nn.Dropout(dropout)

        # Output projection:  [hidden ; context] → vocab
        # hidden_size (current) + hidden_size (attention context) = 2*hidden_size
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Forward pass (teacher-forcing compatible, also usable for generation).

        At each timestep t:
          1. Embed x_t and pass through RNN layers → h_t
          2. Attend over h_1 … h_{t-1}  (all previous hidden states)
          3. Concatenate [h_t ; context] → project to logits

        Parameters
        ----------
        x      : LongTensor (batch, seq_len) — input token indices
        hidden : optional list of hidden states

        Returns
        -------
        logits : Tensor (batch, seq_len, vocab_size)
        hidden : list[Tensor] — final hidden states per layer
        """
        B, T = x.shape
        device = x.device

        # Initialise hidden states to zero if not provided
        if hidden is None:
            hidden = [torch.zeros(B, self.hidden_size, device=device)
                      for _ in range(self.num_layers)]

        emb = self.dropout(self.embedding(x))   # (B, T, E)

        outputs = []
        past_hiddens = []   # accumulate top-layer hidden states for attention

        for t in range(T):
            # ── RNN step ─────────────────────────────────────────────────
            inp = emb[:, t, :]
            new_hidden = []
            for layer, cell in enumerate(self.cells):
                h = cell(inp, hidden[layer])
                inp = self.dropout(h)
                new_hidden.append(h)
            hidden = new_hidden

            current_h = hidden[-1]   # top-layer hidden state at time t

            # ── Self-attention over past hidden states ───────────────────
            if len(past_hiddens) > 0:
                # Stack past hiddens → (B, t, H) and attend
                past = torch.stack(past_hiddens, dim=1)  # (B, num_past, H)
                context, _ = self.attention(past, current_h)
            else:
                # No past states yet (first timestep) — use zeros
                context = torch.zeros(B, self.hidden_size, device=device)

            # Record this timestep's hidden state for future attention
            past_hiddens.append(current_h)

            # ── Combine hidden state + attention context ─────────────────
            combined = torch.cat([current_h, context], dim=-1)  # (B, 2H)
            outputs.append(combined)

        out = torch.stack(outputs, dim=1)   # (B, T, 2H)
        logits = self.fc_out(out)           # (B, T, V)
        return logits, hidden

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
