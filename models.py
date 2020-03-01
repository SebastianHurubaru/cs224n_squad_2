"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


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

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

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
        self.emb = layers.EmbeddingExtra(word_vectors=word_vectors,
                                    args=args)

        self.enc = layers.RNNEncoder(input_size=args.hidden_size,
                                     hidden_size=args.hidden_size,
                                     num_layers=1,
                                     drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.,
                                     extra_size=args.num_features+1)

        self.att = layers.BiDAFAttention(hidden_size=2 * args.hidden_size,
                                         drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else None)

        self.mod = layers.RNNEncoder(input_size=8 * args.hidden_size,
                                     hidden_size=args.hidden_size,
                                     num_layers=2,
                                     drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else 0.)

        self.out = layers.BiDAFOutput(hidden_size=args.hidden_size,
                                      drop_prob=args.drop_prob if hasattr(args, 'drop_prob') else None)

        self.args = args

    def forward(self, cw_idxs, qw_idxs, cw_pos, cw_ner, cw_freq, cqw_extra):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, c_mask, c_len, cw_pos, cw_ner)                                            # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, q_mask, q_len, torch.zeros_like(qw_idxs), torch.zeros_like(qw_idxs))         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(torch.cat([c_emb, cw_freq.contiguous().view(cw_freq.shape[0], cw_freq.shape[1], 1), cqw_extra], 2), c_len)    # (batch_size, c_len, 2 * hidden_size)

        qw_freq = torch.zeros_like(qw_idxs, dtype=torch.float32).contiguous().view(q_emb.shape[0], q_emb.shape[1], 1)
        qcw_extra = torch.full_like(q_emb[:, :, :self.args.num_features], -1)
        q_enc = self.enc(torch.cat([q_emb, qw_freq, qcw_extra], 2), q_len)    # (batch_size, q_len, 2 * hidden_size)

        # c_enc = self.enc(
        #     c_emb,
        #     c_len)  # (batch_size, c_len, 2 * hidden_size)
        #
        # q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class FusionNet(nn.Module):
    """Network for the FusionNet Module."""
    def __init__(self, args, word_vectors):
        super(FusionNet, self).__init__()

        # Input size to RNN: word emb + char emb + question emb + manual features
        input_size = 0

        # Word embeddings
        self.glove = nn.Embedding(nn.Embedding.from_pretrained(word_vectors))

        input_size += args.glove_dim

        # Contextualized embeddings
        self.cove = layers.MTLSTM(args, self.emb)
        input_size += self.cove.output_size

        # POS embeddings
        self.pos_embedding = nn.Embedding(args.pos_size, args.pos_dim)
        input_size += args.pos_dim

        # NER embeddings
        self.ner_embedding = nn.Embedding(args.ner_size, args.ner_dim)
        input_size += args.ner_dim

        # normalized term frequency, match_origin, match_lower, match_lemma
        aux_input = 4

        # Setup the vector size for [premise, hypothesis]
        # they will be modified in the following code
        cur_hidden_size = input_size
        print('Initially, the vector_size is {} (+ {})'.format(cur_hidden_size, aux_input))

        # RNN premise encoder
        self.P_rnn = layers.RNNEncoder(input_size=cur_hidden_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.enc_rnn_layers,
                          drop_prob=args.drop_prob)

        # RNN hypothesis encoder
        self.P_rnn = layers.RNNEncoder(input_size=cur_hidden_size,
                                       hidden_size=args.hidden_size,
                                       num_layers=args.enc_rnn_layers,
                                       drop_prob=args.drop_prob)
        cur_hidden_size = args.hidden_size * 2

        # FA Multi-Level Fusion
        self.full_attn_P = layers.FullAttention(input_size + args.enc_rnn_layers * cur_hidden_size,
                                                args.enc_rnn_layers * cur_hidden_size, args.enc_rnn_layers)
        self.full_attn_H = layers.FullAttention(input_size + args.enc_rnn_layers * cur_hidden_size,
                                                args.enc_rnn_layers * cur_hidden_size, args.enc_rnn_layers)

        cur_hidden_size = self.full_attn_P.output_size * 2

        # RNN premise inference
        self.P_infer_rnn = layers.RNNEncoder(input_size=cur_hidden_size,
                                       hidden_size=args.hidden_size,
                                       num_layers=args.inf_rnn_layers,
                                       drop_prob=args.drop_prob)
        # RNN hypothesis inference
        self.H_infer_rnn = layers.RNNEncoder(input_size=cur_hidden_size,
                                             hidden_size=args.hidden_size,
                                             num_layers=args.inf_rnn_layers,
                                             drop_prob=args.drop_prob)

        cur_hidden_size = args.hidden_size * 2 * args.inf_rnn_layers

        # Question merging
        self.self_attn_P = layers.LinearSelfAttn(cur_hidden_size)
        self.self_attn_H = layers.LinearSelfAttn(cur_hidden_size)

        self.classifier = layers.MLPFunc(cur_hidden_size * 4, cur_hidden_size, args.number_of_class)

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_f, x2_pos, x2_ner, x2_mask):
        """Inputs:
        x1 = premise word indices                [batch * len_1]
        x1_f = premise word features indices     [batch * len_1 * nfeat]
        x1_pos = premise POS tags                [batch * len_1]
        x1_ner = premise entity tags             [batch * len_1]
        x1_mask = premise padding mask           [batch * len_1]
        x2 = hypothesis word indices             [batch * len_2]
        x2_f = hypothesis word features indices  [batch * len_2 * nfeat]
        x2_pos = hypothesis POS tags             [batch * len_2]
        x2_ner = hypothesis entity tags          [batch * len_2]
        x2_mask = hypothesis padding mask        [batch * len_2]
        """
        # Prepare premise and hypothesis input
        Prnn_input_list = []
        Hrnn_input_list = []

        # Word embeddings
        emb = self.embedding if self.training else self.eval_embed
        x1_emb, x2_emb = emb(x1), emb(x2)
        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = layers.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
            x2_emb = layers.dropout(x2_emb, p=self.opt['dropout_emb'], training=self.training)
        Prnn_input_list.append(x1_emb)
        Hrnn_input_list.append(x2_emb)

        # Contextualized embeddings
        _, x1_cove = self.CoVe(x1, x1_mask)
        _, x2_cove = self.CoVe(x2, x2_mask)
        if self.opt['dropout_emb'] > 0:
            x1_cove = layers.dropout(x1_cove, p=self.opt['dropout_emb'], training=self.training)
            x2_cove = layers.dropout(x2_cove, p=self.opt['dropout_emb'], training=self.training)
        Prnn_input_list.append(x1_cove)
        Hrnn_input_list.append(x2_cove)

        # POS embeddings
        x1_pos_emb = self.pos_embedding(x1_pos)
        x2_pos_emb = self.pos_embedding(x2_pos)
        Prnn_input_list.append(x1_pos_emb)
        Hrnn_input_list.append(x2_pos_emb)

        # NER embeddings
        x1_ner_emb = self.ner_embedding(x1_ner)
        x2_ner_emb = self.ner_embedding(x2_ner)
        Prnn_input_list.append(x1_ner_emb)
        Hrnn_input_list.append(x2_ner_emb)

        x1_input = torch.cat(Prnn_input_list, 2)
        x2_input = torch.cat(Hrnn_input_list, 2)

        # Now the features are ready
        # x1_input: [batch_size, doc_len, input_size]
        # x2_input: [batch_size, doc_len, input_size]

        if self.opt['full_att_type'] == 2:
            x1_f = layers.dropout(x1_f, p=self.opt['dropout_EM'], training=self.training)
            x2_f = layers.dropout(x2_f, p=self.opt['dropout_EM'], training=self.training)
            Paux_input, Haux_input = x1_f, x2_f
        else:
            Paux_input = x1_f[:, :, 0].contiguous().view(x1_f.size(0), x1_f.size(1), 1)
            Haux_input = x2_f[:, :, 0].contiguous().view(x2_f.size(0), x2_f.size(1), 1)

        # Encode premise with RNN
        P_abstr_ls = self.P_rnn(x1_input, x1_mask, aux_input=Paux_input)
        # Encode hypothesis with RNN
        H_abstr_ls = self.H_rnn(x2_input, x2_mask, aux_input=Haux_input)

        # Fusion
        if self.opt['full_att_type'] == 0:
            P_atts = P_abstr_ls[-1].contiguous()
            H_atts = H_abstr_ls[-1].contiguous()
            P_xs = P_abstr_ls[-1].contiguous()
            H_xs = H_abstr_ls[-1].contiguous()
        elif self.opt['full_att_type'] == 1:
            P_atts = torch.cat([x1_input] + P_abstr_ls, 2)
            H_atts = torch.cat([x2_input] + H_abstr_ls, 2)
            P_xs = P_abstr_ls[-1].contiguous()
            H_xs = H_abstr_ls[-1].contiguous()
        elif self.opt['full_att_type'] == 2:
            P_atts = torch.cat([x1_input] + P_abstr_ls, 2)
            H_atts = torch.cat([x2_input] + H_abstr_ls, 2)
            P_xs = torch.cat(P_abstr_ls, 2)
            H_xs = torch.cat(H_abstr_ls, 2)
        aP_xs = self.full_attn_P(P_atts, H_atts, P_xs, H_xs, x2_mask)
        aH_xs = self.full_attn_H(H_atts, P_atts, H_xs, P_xs, x1_mask)
        P_hiddens = torch.cat([P_xs, aP_xs], 2)
        H_hiddens = torch.cat([H_xs, aH_xs], 2)

        # Inference on premise and hypothesis
        P_hiddens = torch.cat(self.P_infer_rnn(P_hiddens, x1_mask), 2)
        H_hiddens = torch.cat(self.H_infer_rnn(H_hiddens, x2_mask), 2)

        # Merge hiddens for answer classification
        if self.opt['final_merge'] == 'avg':
            P_merge_weights = layers.uniform_weights(P_hiddens, x1_mask)
            H_merge_weights = layers.uniform_weights(H_hiddens, x2_mask)
        elif self.opt['final_merge'] == 'linear_self_attn':
            P_merge_weights = self.self_attn_P(P_hiddens, x1_mask)
            H_merge_weights = self.self_attn_H(H_hiddens, x2_mask)
        P_avg_hidden = layers.weighted_avg(P_hiddens, P_merge_weights)
        H_avg_hidden = layers.weighted_avg(H_hiddens, H_merge_weights)
        P_max_hidden = torch.max(P_hiddens, 1)[0]
        H_max_hidden = torch.max(H_hiddens, 1)[0]

        # Predict scores for different classes
        scores = self.classifier(torch.cat([P_avg_hidden, H_avg_hidden, P_max_hidden, H_max_hidden], 1))

        return scores # -inf to inf