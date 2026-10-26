"""
Шаг 2. Нейросетевые модели на PyTorch:
1. RecGPT2 — генеративная модель на архитектуре GPT-2 (Causal Transformer), предсказывающая следующий фильм по истории.
2. RewardModel — модель вознаграждения, предсказывающая отклик пользователя на фильм (доля просмотра / удержание).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecGPT2(nn.Module):
    """
    Архитектура GPT-2 для рекомендаций (Next-Item Prediction).
    Токены — это ID фильмов (1..vocab_size), 0 — токен паддинга (PAD).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        # Выходная голова предсказывает логиты для всех фильмов 1..vocab_size
        self.head = nn.Linear(embed_dim, vocab_size + 1, bias=False)
        # Связываем веса эмбеддингов и выходного слоя (Weight Tying, как в оригинальном GPT-2)
        self.head.weight = self.token_emb.weight

    def get_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """Возвращает скрытые состояния трансформера с каузальной маской (чтобы не подглядывать вперёд)."""
        b, t = seq.shape
        positions = torch.arange(0, t, device=seq.device).unsqueeze(0).expand(b, t)
        x = self.drop(self.token_emb(seq) + self.pos_emb(positions))

        # Каузальная маска GPT-2: позиция i видит только позиции <= i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=seq.device)
        x = self.blocks(x, mask=causal_mask)
        return self.ln_f(x)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты следующего объекта для каждой позиции последовательности."""
        h = self.get_hidden(seq)
        logits = self.head(h)
        logits[:, :, 0] = -1e9  # Запрещаем предсказывать токен паддинга (0)
        return logits


class RewardModel(nn.Module):
    """
    Модель вознаграждения (Reward Model).
    Принимает историю пользователя и рекомендованный фильм (candidate_item),
    возвращает ожидаемую награду r in [0, 1] (например, ожидаемую долю досмотра фильма).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context_seq: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        """
        context_seq: [B, T] — история просмотренных объектов.
        candidate_items: [B] или [B, K] — кандидаты для оценки вознаграждения.
        """
        emb_seq = self.item_emb(context_seq)
        _, h_last = self.gru(emb_seq)
        user_state = h_last.squeeze(0)  # [B, D]

        if candidate_items.dim() == 1:
            cand_emb = self.item_emb(candidate_items)  # [B, D]
            feats = torch.cat([user_state, cand_emb, user_state * cand_emb], dim=-1)
            return self.mlp(feats).squeeze(-1)
        else:
            # Оценка сразу K кандидатов для каждого пользователя: candidate_items [B, K]
            b, k = candidate_items.shape
            cand_emb = self.item_emb(candidate_items)  # [B, K, D]
            u_exp = user_state.unsqueeze(1).expand(b, k, -1)
            feats = torch.cat([u_exp, cand_emb, u_exp * cand_emb], dim=-1)
            return self.mlp(feats).squeeze(-1)  # [B, K]
