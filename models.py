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
    """
    Network for the FusionNet Module.
    """

    def __init__(self, args, word_vectors):
        super(FusionNet, self).__init__()

        self.args = args

        # Input size to RNN: word emb + char emb + question emb + manual features
        input_size = 0

        # Word embeddings
        self.glove = nn.Embedding.from_pretrained(word_vectors)

        input_size += args.glove_dim

        # Contextualized embeddings
        self.cove = layers.MT_LSTM(args, word_vectors)
        input_size += self.cove.output_size

        # POS embeddings
        self.pos_embedding = nn.Embedding(args.pos_size, args.pos_dim)

        # NER embeddings
        self.ner_embedding = nn.Embedding(args.ner_size, args.ner_dim)

        # normalized term frequency, match_origin, match_lower, match_lemma
        extra_feat_dim = args.num_features + 1

        # Fully-Aware Multi-level Fusion: Word-level
        # In our initial computation of the model the Fully-Aware Multi-level Fusion was set at a level one, however
        # performance is likely to become better by having multi-level fusion (>1),
        # we experimented by increasing to 6 levels and are currently assessing the performance
        # we will also try with 3 levels at the next step to compare between 1 level, 3 levels and 6 levels.
        # performance with 6 levels was worse than 1 (appearing to come from overfitting), trying with 2 instead for the next test
        self.full_attn_word_level = layers.FullyAwareAttention(args, args.glove_dim, args.glove_dim, 1)

        # Reading
        self.reading_context = layers.MultiLevelRNNEncoder(args=args,
                                                           input_size=input_size + args.pos_dim + args.ner_dim + extra_feat_dim + self.full_attn_word_level.output_size,
                                                           hidden_size=args.concepts_size,
                                                           num_layers=args.enc_rnn_layers)

        self.reading_question = layers.MultiLevelRNNEncoder(args=args,
                                                            input_size=input_size,
                                                            hidden_size=args.concepts_size,
                                                            num_layers=args.enc_rnn_layers)

        # Question understanding
        self.final_ques = layers.MultiLevelRNNEncoder(args=args,
                                                      input_size=2 * args.concepts_size * args.enc_rnn_layers,
                                                      hidden_size=args.concepts_size,
                                                      num_layers=1)

        # FA Multi-Level Fusion
        self.full_attn_low_level = layers.FullyAwareAttention(args,
                                                              input_size + 2 * args.concepts_size * args.enc_rnn_layers,
                                                              args.concepts_size * args.enc_rnn_layers,
                                                              args.enc_rnn_layers)
        self.full_attn_high_level = layers.FullyAwareAttention(args,
                                                               input_size + 2 * args.concepts_size * args.enc_rnn_layers,
                                                               args.concepts_size * args.enc_rnn_layers,
                                                               args.enc_rnn_layers)
        self.full_attn_understand = layers.FullyAwareAttention(args,
                                                               input_size + 2 * args.concepts_size * args.enc_rnn_layers,
                                                               args.concepts_size * args.enc_rnn_layers,
                                                               args.enc_rnn_layers)

        self.fully_focused_context = layers.MultiLevelRNNEncoder(args=args,
                                                                 input_size=2 * args.concepts_size * 5,
                                                                 hidden_size=args.concepts_size,
                                                                 num_layers=1)

        self.full_attn_history_of_word = layers.FullyAwareAttention(args,
                                                                    input_size + 2 * args.concepts_size * 6,
                                                                    args.concepts_size * args.enc_rnn_layers,
                                                                    args.enc_rnn_layers)

        self.final_context = layers.MultiLevelRNNEncoder(args=args,
                                                         input_size=2 * args.concepts_size * args.enc_rnn_layers,
                                                         hidden_size=args.concepts_size,
                                                         num_layers=1)

        # Output
        self.summarized_final_ques = layers.LinearSelfAttention(args=args,
                                                                input_size=2 * args.concepts_size)

        self.span_start = layers.LinearSelfAttention(args=args,
                                                     input_size=2 * args.concepts_size,
                                                     hidden_size=2 * args.concepts_size,
                                                     is_output=True)

        self.combine_context_span_start_ques_under = nn.GRU(input_size=2 * args.concepts_size,
                                                            hidden_size=2 * args.concepts_size,
                                                            batch_first=True,
                                                            bidirectional=False)

        self.span_end = layers.LinearSelfAttention(args=args,
                                                   input_size=2 * args.concepts_size,
                                                   hidden_size=2 * args.concepts_size,
                                                   is_output=True)

    def forward(self, cw_idxs, qw_idxs, cw_pos, cw_ner, cw_freq, cqw_extra):
        """
        Inputs:
        cw_idxs: tensor representing for each batch the indexes for the words of the context
            with size torch.size([batch_size, length of the context in words for the longest context in the batch])
        qw_idxs: tensor representing for each batch the indexes for the words of the question
            with torch.size([batch_size, length of the question in words for the longest question in the batch])
        cw_pos: 12-dim part-of-speech (POS),
            with torch.size([batch_size, length of the context in words for the longest context in the batch])
        cw_ner: 8-dim Name Entity Recognition (NER) embedding,
            with torch.size([batch_size, length of the context in words for the longest context in the batch])
        cw_freq: Normalized term frequency (TF) for context C,
            with torch.size([batch_size, length of the context in words for the longest context in the batch])
        cqw_extra:
            with torch.size([batch_size, length of the context in words for the longest context in the batch], 3)


        Outputs:
        P_s: Token start position for each batch

        P_e: Token end position for each batch
        """

        # c_mask is a tensor with the same size as cw_idxs
        # the value of items in the tensor are equal to 0 if the value
        # for the item of cw_idxs are equal to 0 and 1 otherwise:
        # a value of 0 for an item of c_idxs indicates padding
        # for example, with a theoretical example of cw_idxs = torch([0,0,4,5,6,0])
        # the value of  c_mask would be torch([0,0,1,1,1,0])
        # Please note that 0 is use for padding in cw_idx and indicates that there is no word.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # q_mask is a tensor with the same size as cw_idxs,
        # the value of items in the tensor are equal to 0 if the values
        # for the item of qw_idxs are equal to 0 and 1 otherwise:
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        # The masks for the context and the questions are summed up on their last dimensions,
        # which gives for each batch respectively the length of the context (c_len) and length of the question (q_len)
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # Negate the mask, i.e. 1 if padded
        c_mask = ~c_mask
        q_mask = ~q_mask

        # Word embeddings
        # Each word in C and Q is transformed into an input vector using the 300-dim GloVe embedding
        # (Pennington et al. 2014) by creating an embedding instance (by calling self.glove,
        # which is nn.Embedding.from_pretrained(word_vectors)
        g_c, g_q = self.glove(cw_idxs), self.glove(qw_idxs) # with torch.size([batch_size, length of the context in words for the longest context in the batch], 300)
                                                            # with torch.size([batch_size, length of the question in words for the longest question in the batch], 300)
        # Dropout on embeddings
        if self.args.drop_prob > 0:
            g_c = F.dropout(g_c, p=self.args.drop_prob, training=self.training)
            g_q = F.dropout(g_q, p=self.args.drop_prob, training=self.training)

        # Contextualized embeddings use as parameters the indexes of the words of the context/question and the
        # number of words in the context/question, and call a Multi-Timescale LSTM
        # further reference on Multi-Timescale LSTM can be found in Learned in Translation: Contextualized Word Vectors
        # at https://arxiv.org/pdf/1708.00107.pdf
        # We feed GloVe(cw_idxs) and GloVe(qw_idxs) to a standard, two-layer, bidirectional, long short-term memory network
        # (cf. Gravesand Schmidhuber, 2005) that we refer to as an MT-LSTM to indicate that it is this same two-layer BiLSTM
        # that we later transfer as a pretrained encoder. The MT-LSTM is used to compute a sequence of hidden states
        # MT-LSTM(GloVe(cw_idxs)) and MT-LSTM(GloVe(qw_idxs)).
        # used as embeddings
        _, c_c = self.cove(cw_idxs, c_len)  # with torch.size([batch_size, length of the context in words for the longest context in the batch], 600)
        _, c_q = self.cove(qw_idxs, q_len)  # with torch.size([batch_size, length of the question in words for the longest question in the batch], 600)
        if self.args.drop_prob > 0:
            c_c = F.dropout(c_c, p=self.args.drop_prob, training=self.training)
            c_q = F.dropout(c_q, p=self.args.drop_prob, training=self.training)

        # part-of-speech (POS) embeddings: as descrived on page 5 of the FusionNet paper (Huang et al.), in the SQuaD task
        # we also include 12 dim POS embedding
        # See also: https://arxiv.org/pdf/1704.00051.pdf Reading Wikipedia to Answer Open-Domain Questions
        c_pos_emb = self.pos_embedding(cw_pos) # with torch.size([batch_size, length of the context in words for the longest context in the batch], 12)

        # named entity recognition (NER) embeddings
        c_ner_emb = self.ner_embedding(cw_ner) # with torch.size([batch_size, length of the context in words for the longest context in the batch], 8)


        # Fully-Aware Multi-level Fusion: Word-level
        # Note that the parameter q_mask is equal to 1 if padded and 0 otherwise.
        # self.full_attn_word_level calls layers.FullyAwareAttention
        # In multi-level fusion, we separetely consider fusing word-level and higher level. Word-level fusion informs C
        # about what kind of words are in Q. For this component, we follow the approach of Chen et al. 2017a.
        # In the initial test of the multi-level fusion, we used a parameter 1, meaning one level only
        # in the the next tests we also used 6 (waiting for result) and also intending to use 3 levels of fusions

        g_hat_c = self.full_attn_word_level(g_c, g_q,
                                            g_q, q_mask) #with torch.size([batch_size, length of the context in words for the longest context in the batch], 300)

        # Creation of input vectors w_c and w_q (page 5 of fusionnet article).
        # ---------------------------------------------------------------------
        # Create vector an input vector w_c by concatening g_c, c_c, c_pos_emb, c_ner_emb, cw_freq.unsqueeze(-1)
        # g_c, c_c, c_pos_emb, c_ner_emb have the same dimensions at 0 and 1.
        # cw_freq has dimension torch.Size([64, 305]), and is converted to dimension torch.Size([64, 305,1])
        # using the command cw_freq.unsqueeze(-1)
        # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        # unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.
        # The returned tensor shares the same underlying data with this tensor.
        w_c = torch.cat([g_c, c_c, c_pos_emb, c_ner_emb, cw_freq.unsqueeze(-1)], 2)
        # create an input vector w_q by concatening g_q, c_q along the dimension 2
        # therefore concatenate two vectors: g_q of size torch.Size([batch size, max nb words question, 300]) and
        # torch.Size([batch size, max nb words question,600]) into a vector w_q of size
        # torch.Size([batch size, max nb words question, 900])
        w_q = torch.cat([g_q, c_q], 2)

        # Reading (page 6 of FusionNet article)
        # In the reading component, we use a separate bidirectional LSTm (BiLSTM) calling layers.MultiLevelRNNEncoder
        # to form low-level and high-level concepts for C and Q. Hence low-level and high-level concepts are created
        # for each word.
        h_c_l, h_c_h = self.reading_context(torch.cat([w_c, cqw_extra, g_hat_c], 2),
                                            c_mask)

        h_q_l, h_q_h = self.reading_question(w_q, q_mask)


        # Understanding vector for the question
        # Question Understanding: In the Question Understanding component, we apply a new BiLSTM
        # layers.MultiLevelRNNEncoder taking in both h_q_l and h_q_h to obtain the final question representation U_q:
        U_q = self.final_ques(torch.cat([h_q_l, h_q_h], 2),
                              q_mask)[0]

        # Fully-Aware Multi-level Fusion: Higher-level. Explanation on page 6 of the FusionNet article
        # This component fuses all higher-level informationin the question Q to the context C through fully-aware attention
        # on history-of-word.   Since the proposed attention scoring function for fully-aware attention is constrained
        # to be symmetric,  we need to identify the common history-of-word for both C,Q. This achieved by
        # concatenating the the sequences of  tensors in the given dimension.
        # All tensors have the same shape (except in the concatenating dimension).
        # g_c and g_q are the GloVe embeddings and c_c and c_q are the CoVe embedding
        # h_c_l, h_c_h, h_q_l and h_q_h are the low level and high level concepts for C and Q generated by the
        # 4 BiLSTM.
        HoW_c = torch.cat([g_c, c_c, h_c_l, h_c_h], 2)
        HoW_q = torch.cat([g_q, c_q, h_q_l, h_q_h], 2)

        # Using the history of words for the context HoW_c and for the question HoW_q, we compute the
        # (i) low-level fusion, (ii) high-level fusion and (iii) understanding fusion using Fully Aware Attention.
        # The difference in input between the 3 fusions is the use of h_q_l (low-level), h_q_h (high-level) and
        # U_q (understanding).


        # Low-level fusion (equation C1 in Fusionnet page 6)
        # h_hat_c_l = \sum_j alpha^l_{ij}  h_q_l_{j} with alpha^l_{ij} proportional to an attention function^l (HoW_c_i, HoW_q_j)
        h_hat_c_l = self.full_attn_low_level(HoW_c, HoW_q,
                                             h_q_l, q_mask)

        # High-level fusion (equation C1 in Fusionnet page 6)
        # h_hat_c_h = \sum_j alpha^h_{ij}  h_q_h_{j} with alpha^h_{ij} proportional to an attention function^h (HoW_c_i, HoW_q_j)
        h_hat_c_h = self.full_attn_high_level(HoW_c, HoW_q,
                                              h_q_h, q_mask)

        # Understanding fusion (equation C1 in Fusionnet page 6)
        # h_hat_c = \sum_j alpha^u_{ij}  U_q_{j} with alpha^h_{ij} proportional to an attention function^u (HoW_c_i, HoW_q_j)
        u_hat_c = self.full_attn_understand(HoW_c, HoW_q,
                                            U_q, q_mask)

        # We concatenate the tensors h_c_l, h_c_h, h_hat_c_l, h_hat_c_h, u_hat_c over the dimension 2 and pass it
        # as first parameter to MultiLevelRNNEncoder with the second parameter being c_mask
        # As a reminder c_mask is equal to 1 if padded and 0 otherwise.
        # This new BiLSTM is applied to obtain the representation for C fully fused with information in the question Q.
        # (equation C2 in Fusionnet page 6)
        v_c = self.fully_focused_context(torch.cat([h_c_l, h_c_h, h_hat_c_l, h_hat_c_h, u_hat_c], 2),
                                         c_mask)[0]

        # Fully-Aware Self-Boosted Fusion

        # Define the new history of word by concatening the tensors g_c, c_c, h_c_l, h_c_h, h_hat_c_l, h_hat_c_h,
        # u_hat_c, v_c along dimension 2:
        HoW_c = torch.cat([g_c, c_c, h_c_l, h_c_h, h_hat_c_l, h_hat_c_h, u_hat_c, v_c], 2)

        # The fully aware attention v_hat_c is computed through Fully Aware Attention
        # (equation C3 of FusionNet article page 6).
        v_hat_c = self.full_attn_history_of_word(HoW_c, HoW_c,
                                                 v_c, c_mask)

        # Understanding vector for the context
        # The Final context representation U_c represents the understanding vector for the context C, which are fully
        # fused with with the question Q.
        #  U_c is obtained by applying a BiLSTM (MultiLevelRNNEncoder) on the
        # concatenation of the tensors v_c, v_hat_c (equation C4 of the FusionNet article on page 6)

        U_c = self.final_context(torch.cat([v_c, v_hat_c], 2), 1)[0]


        # The output of FusionNet are the understanding vectors U_c and U_q for both C and Q.


        # Computation of the answer span in the context:

        # Summarized question understanding vector
        # The single summarized question understanding vector u_q is obtained by computing
        # \sum_i \beta_i U_q_{i), where \beta_i is proportional to \exp(w^T u_i^Q) and w is a trainable vector

        u_q = self.summarized_final_ques(None, U_q, q_mask)

        # The span start P_s is computed using the summarized question understanding vector u_q
        # [Need to explain how it is computed by checking the code in layers.py]
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        P_s = self.span_start(u_q, U_c, c_mask)

        combine = U_c.transpose(1, 2).bmm(torch.exp(P_s.unsqueeze(-1))).squeeze(-1)

        combine = F.dropout(combine, self.args.drop_prob, self.training)
        self.combine_context_span_start_ques_under.flatten_parameters()
        v_q, _ = self.combine_context_span_start_ques_under(combine.unsqueeze(1), u_q.unsqueeze(0))

        P_e = self.span_end(v_q.squeeze(1), U_c, c_mask)

        return P_s, P_e
