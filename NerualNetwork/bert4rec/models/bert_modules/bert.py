from torch import nn as nn
import torch
from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        self.is_original = args.original
        max_len = args.bert_max_len
        self.max_len = max_len
        num_items = args.bert_num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        # 2 means [mask] (item_num + 1) and padding (0)
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        hidden_dropout = args.bert_hidden_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden,
                                       max_len=max_len, dropout=hidden_dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout, hidden_dropout) for _ in range(n_layers)])

        self.unidirectional_tf_blocks = None

    def forward(self, x, test):
        if self.is_original == 1:
            # x's dimension: batch_size x max_len
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            # batch_size x 1 x max_len x max_len
            x = self.embedding(x)
            for transformer in self.transformer_blocks:
                x = transformer.forward(x, mask)
            return x
        else:
            # x's dimension: batch_size x max_len
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            # batch_size x 1 x max_len x max_len

            mask_backward = torch.tril(torch.ones(self.max_len, self.max_len), diagonal=-1)

            mask_backward = mask_backward.unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(1)

            mask_backward = mask_backward * mask

            # embedding the indexed sequence to sequence of vectors
            x = self.embedding(x)
            y = x.clone()

            # print("embedding size: ", x.size())

            # running over multiple transformer blocks
            for transformer in self.transformer_blocks:
                x = transformer.forward(x, mask)

            if not test:
                for transformer in self.transformer_blocks:
                    y = transformer.forward(y, mask_backward)

            # print("output size of bert: ", x.size())
            return (x, y) if not test else x# dimension: Batch_size x max_item_size x hidden_size

    def init_weights(self):
        pass
