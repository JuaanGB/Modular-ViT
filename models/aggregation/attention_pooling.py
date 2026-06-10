import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregation.base import BaseAggregation
from models.ExecutionState import ExecutionState


class AttentionPoolingAggregation(BaseAggregation):
    """
    Attention Pooling:
    y = sum_i alpha_i x_i
    alpha_i = softmax(MLP(x_i))
    """

    def __init__(
        self,
        has_cls_token: bool = False,
        hidden_dim: int = None,
        execution_state: ExecutionState = None
    ):
        super().__init__(has_cls_token, execution_state)

        embed_dim = execution_state.embed_dim
        hidden_dim = hidden_dim or embed_dim

        # Scoring network (MLP)
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        returns: [B, D]
        """

        # Si hay CLS token, lo ignoramos (opcional pero recomendable para comparar con GAP)
        if self.has_cls_token:
            x_tokens = x[:, 1:, :]
        else:
            x_tokens = x

        # scores: [B, N, 1]
        scores = self.score_net(x_tokens)
 
        # attention weights: [B, N, 1]
        weights = torch.softmax(scores, dim=1)

        # weighted sum
        # [B, N, 1] * [B, N, D] -> [B, N, D]
        weighted = weights * x_tokens

        # sum over tokens -> [B, D]
        out = weighted.sum(dim=1)

        return out

    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseAggregation:
        return AttentionPoolingAggregation(
            has_cls_token=config.get("has_cls_token", False),
            hidden_dim=config.get("hidden_dim", None),
            execution_state=execution_state
        )