"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFExtra(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, args):
        super(BiDAFExtra, self).__init__()

        self.c_emb = layers.EmbeddingExtra(word_vectors=word_vectors,
                                           args=args,
                                           aux_feat=True)

        self.q_emb = layers.EmbeddingExtra(word_vectors=word_vectors,
                                           args=args,
                                           aux_feat=False)

        self.c_enc = layers.RNNEncoder(input_size=args.hidden_size + args.num_features,
                                       hidden_size=args.hidden_size,
                                       num_layers=1,
                                       drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.)

        self.q_enc = layers.RNNEncoder(input_size=args.hidden_size,
                                       hidden_size=args.hidden_size,
                                       num_layers=1,
                                       drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.)

        self.att = layers.BiDAFAttention(hidden_size=2 * args.hidden_size,
                                         drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.)

        self.mod = layers.RNNEncoder(input_size=8 * args.hidden_size,
                                     hidden_size=args.hidden_size,
                                     num_layers=2,
                                     drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.)

        self.out = layers.BiDAFOutput(hidden_size=args.hidden_size,
                                      drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.)

        self.args = args

    def forward(self, cw_idxs, qw_idxs, cw_pos, cw_ner, cw_freq, cqw_extra):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.c_emb(cw_idxs, c_mask, c_len, cw_pos, cw_ner, cw_freq)
        q_emb = self.q_emb(qw_idxs, q_mask, q_len, None, None, None)

        c_enc = self.c_enc(
            torch.cat([c_emb, cqw_extra], 2),
            c_len)

        q_enc = self.q_enc(q_emb, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)

        mod = self.mod(att, c_len)

        out = self.out(att, mod, c_mask)

        return out


class FusionNet(nn.Module):
    """Network for the FusionNet Module."""

    def __init__(self, args, word_vectors):
        super(FusionNet, self).__init__()

        self.args = args

        # Input size to RNN: word emb + char emb + question emb + manual features
        input_size = 0

        # Word embeddings
        self.glove = nn.Embedding.from_pretrained(word_vectors)

        input_size += args.glove_dim

        # Contextualized embeddings
        self.cove = layers.MTLSTM(args, word_vectors)
        input_size += self.cove.output_size

        # POS embeddings
        self.pos_embedding = nn.Embedding(args.pos_size, args.pos_dim)

        # NER embeddings
        self.ner_embedding = nn.Embedding(args.ner_size, args.ner_dim)

        # normalized term frequency, match_origin, match_lower, match_lemma
        extra_feat_dim = args.num_features + 1

        # Fully-Aware Multi-level Fusion: Word-level
        self.full_attn_word_level = layers.FullAttention(args, args.glove_dim, args.glove_dim, 1)

        # Reading
        self.reading_context = layers.FNRNNEncoder(args=args,
                                                   input_size=input_size + args.pos_dim + args.ner_dim + extra_feat_dim + self.full_attn_word_level.output_size,
                                                   hidden_size=args.concepts_size,
                                                   num_layers=args.enc_rnn_layers)

        self.reading_question = layers.FNRNNEncoder(args=args,
                                                    input_size=input_size,
                                                    hidden_size=args.concepts_size,
                                                    num_layers=args.enc_rnn_layers)

        # Question understanding
        self.final_ques = layers.FNRNNEncoder(args=args,
                                              input_size=2 * args.concepts_size * args.enc_rnn_layers,
                                              hidden_size=args.concepts_size,
                                              num_layers=1)

        # FA Multi-Level Fusion
        self.full_attn_low_level = layers.FullAttention(args,
                                                        input_size + 2 * args.concepts_size * args.enc_rnn_layers,
                                                        args.concepts_size * args.enc_rnn_layers,
                                                        args.enc_rnn_layers)
        self.full_attn_high_level = layers.FullAttention(args,
                                                         input_size + 2 * args.concepts_size * args.enc_rnn_layers,
                                                         args.concepts_size * args.enc_rnn_layers,
                                                         args.enc_rnn_layers)
        self.full_attn_understand = layers.FullAttention(args,
                                                         input_size + 2 * args.concepts_size * args.enc_rnn_layers,
                                                         args.concepts_size * args.enc_rnn_layers,
                                                         args.enc_rnn_layers)

        self.fully_focused_context = layers.FNRNNEncoder(args=args,
                                                         input_size=2 * args.concepts_size * 5,
                                                         hidden_size=args.concepts_size,
                                                         num_layers=1)

        self.full_attn_history_of_word = layers.FullAttention(args,
                                                              input_size + 2 * args.concepts_size * 6,
                                                              args.concepts_size * args.enc_rnn_layers,
                                                              args.enc_rnn_layers)

        self.final_context = layers.FNRNNEncoder(args=args,
                                                 input_size=2 * args.concepts_size * args.enc_rnn_layers,
                                                 hidden_size=args.concepts_size,
                                                 num_layers=1)

        # Output
        self.summarized_final_ques = layers.LinearSelfAttn(args=args,
                                                           input_size=2 * args.concepts_size)

        self.span_start = layers.LinearSelfAttn(args=args,
                                                input_size=2 * args.concepts_size,
                                                hidden_size=2 * args.concepts_size,
                                                is_output=True)

        self.combine_context_span_start_ques_under = layers.FNRNNEncoder(args=args,
                                                                         input_size=2 * args.concepts_size * args.enc_rnn_layers,
                                                                         hidden_size=args.concepts_size,
                                                                         num_layers=1,
                                                                         rnn_type=nn.GRU)

        self.span_end = layers.LinearSelfAttn(args=args,
                                              input_size=2 * args.concepts_size,
                                              hidden_size=2 * args.concepts_size,
                                              is_output=True)

    def forward(self, cw_idxs, qw_idxs, cw_pos, cw_ner, cw_freq, cqw_extra):
        """Inputs:

        """

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # Negate the mask, i.e. 1 if padded
        c_mask = ~c_mask
        q_mask = ~q_mask

        # Word embeddings
        g_c, g_q = self.glove(cw_idxs), self.glove(qw_idxs)

        # Dropout on embeddings
        if self.args.drop_prob > 0:
            g_c = F.dropout(g_c, p=self.args.drop_prob, training=self.training)
            g_q = F.dropout(g_q, p=self.args.drop_prob, training=self.training)

        # Contextualized embeddings
        _, c_c = self.cove(cw_idxs, c_len)
        _, c_q = self.cove(qw_idxs, q_len)
        if self.args.drop_prob > 0:
            c_c = F.dropout(c_c, p=self.args.drop_prob, training=self.training)
            c_q = F.dropout(c_q, p=self.args.drop_prob, training=self.training)

        # POS embeddings
        c_pos_emb = self.pos_embedding(cw_pos)

        # NER embeddings
        c_ner_emb = self.ner_embedding(cw_ner)

        # Fully-Aware Multi-level Fusion: Word-level
        g_hat_c = self.full_attn_word_level(g_c, g_q,
                                            g_q, q_mask)

        # Reading
        w_c = torch.cat([g_c, c_c, c_pos_emb, c_ner_emb, cw_freq.unsqueeze(-1)], 2)
        w_q = torch.cat([g_q, c_q], 2)

        h_c_l, h_c_h = self.reading_context(torch.cat([w_c, cqw_extra, g_hat_c], 2),
                                            c_mask)

        h_q_l, h_q_h = self.reading_question(w_q, q_mask)

        # Question Understanding
        U_q = self.final_ques(torch.cat([h_q_l, h_q_h], 2),
                              q_mask)[0]

        # Fully-Aware Multi-level Fusion: Higher-level.
        HoW_c = torch.cat([g_c, c_c, h_c_l, h_c_h], 2)
        HoW_q = torch.cat([g_q, c_q, h_q_l, h_q_h], 2)

        # Low-level fusion
        h_hat_c_l = self.full_attn_low_level(HoW_c, HoW_q,
                                             h_q_l, q_mask)

        # Low-level fusion
        h_hat_c_h = self.full_attn_high_level(HoW_c, HoW_q,
                                              h_q_h, q_mask)

        # Understanding fusion
        u_hat_c = self.full_attn_understand(HoW_c, HoW_q,
                                            U_q, q_mask)

        v_c = self.fully_focused_context(torch.cat([h_c_l, h_c_h, h_hat_c_l, h_hat_c_h, u_hat_c], 2),
                                         c_mask)[0]

        # Fully-Aware Self-Boosted Fusion
        HoW_c = torch.cat([g_c, c_c, h_c_l, h_c_h, h_hat_c_l, h_hat_c_h, u_hat_c, v_c], 2)

        v_hat_c = self.full_attn_history_of_word(HoW_c, HoW_c,
                                                 v_c, c_mask)

        U_c = self.final_context(torch.cat([v_c, v_hat_c], 2), 1)[0]

        # Output
        u_q = self.summarized_final_ques(None, U_q, q_mask)

        P_s = self.span_start(u_q, U_c, c_mask)

        combine = U_c.transpose(1, 2).bmm(torch.exp(P_s.unsqueeze(-1))).squeeze(-1)

        v_q = self.combine_context_span_start_ques_under(torch.cat([combine, u_q], 1).unsqueeze(1), 1)[0]

        P_e = self.span_end(v_q.squeeze(1), U_c, c_mask)

        return P_s, P_e
