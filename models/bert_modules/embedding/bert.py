import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embedding_size, output_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embedding_size
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=self.embed_size)


        self.position = PositionalEmbedding(max_len=max_len, d_model=self.embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.mat = nn.Linear(self.embed_size, output_size)

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)

        return self.dropout(self.mat(x))
