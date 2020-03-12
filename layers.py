"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)

        return emb


class EmbeddingExtra(nn.Module):

    def __init__(self, args, word_vectors, aux_feat=True):

        """Embedding layer used by BiDAFExtra, without the character-level component.

    Word-level embeddings are extended with Contextual Word Vectors, POS tags, NER and word frequency.
    Further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        args: arguments passed to the main program
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.

    """
        super(EmbeddingExtra, self).__init__()

        self.args = args
        self.aux_feat = aux_feat

        self.drop_prob = args.drop_prob if hasattr(args, 'drop_prob') else 0.
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        input_size = args.glove_dim

        self.cove = MTLSTM(args, word_vectors)
        input_size += args.cove_dim

        if self.aux_feat is True:

            # POS embeddings
            self.pos_embed = nn.Embedding(args.pos_size, args.pos_dim)
            input_size += args.pos_dim

            # NER embeddings
            self.ner_embed = nn.Embedding(args.ner_size, args.ner_dim)
            input_size += args.ner_dim

            # Word frequency
            input_size += 1

        self.proj = nn.Linear(input_size, args.hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, args.hidden_size)

    def forward(self, x, x_mask, lengths, x_pos, x_ner, x_freq):
        input_list = []

        glove_emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        glove_emb = F.dropout(glove_emb, self.drop_prob, self.training)
        input_list.append(glove_emb)

        _, cove_emb = self.cove(x, lengths)
        cove_emb = F.dropout(cove_emb, self.drop_prob, self.training)
        input_list.append(cove_emb)

        if self.aux_feat is True:

            pos_emb = self.pos_embed(x_pos)
            input_list.append(pos_emb)

            ner_emb = self.ner_embed(x_ner)
            input_list.append(ner_emb)

            input_list.append(x_freq.unsqueeze(-1))

        inputs = torch.cat(input_list, 2)

        emb = self.proj(inputs)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class FNRNNEncoder(nn.Module):

    def __init__(self, args, input_size, hidden_size, num_layers, rnn_type=nn.LSTM):
        super(FNRNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size_ = (input_size + 2 * hidden_size * i)
            self.rnns.append(rnn_type(input_size_, hidden_size, num_layers=1, bidirectional=True))

        self.args = args

    def forward(self, x, x_mask):
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = torch.cat(hiddens, 2)

            # Apply dropout to input
            if self.args.drop_prob > 0:
                rnn_input = F.dropout(rnn_input, self.args.drop_prob, self.training)

            # Forward
            self.rnns[i].flatten_parameters()
            rnn_output = self.rnns[i](rnn_input)[0]
            hiddens.append(rnn_output)

        # Transpose back
        hiddens = [h.transpose(0, 1) for h in hiddens]
        return hiddens[1:]


class MTLSTM(nn.Module):

    def __init__(self, args, word_vectors):
        """Initialize an Multi-Timescale LSTM

        Arguments:

        """
        super(MTLSTM, self).__init__()

        self.embed = nn.Embedding.from_pretrained(word_vectors)

        # state_dict = torch.load(opt['MTLSTM_path'])
        self.rnn1 = nn.LSTM(args.glove_dim, args.glove_dim, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(args.cove_dim, args.glove_dim, num_layers=1, bidirectional=True)

        state_dict = torch.load(args.cove_emb_file)
        state_dict1 = dict([(name, param) for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param) for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)

        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.rnn1.parameters():
            p.requires_grad = False
        for p in self.rnn2.parameters():
            p.requires_grad = False

        self.output_size = args.cove_dim

    def forward(self, x, lengths):

        x = self.embed(x)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNNs
        self.rnn1.flatten_parameters()
        output1, _ = self.rnn1(x)

        self.rnn2.flatten_parameters()
        output2, _ = self.rnn2(output1)

        output1, _ = pad_packed_sequence(output1, batch_first=True)
        output2, _ = pad_packed_sequence(output2, batch_first=True)

        # Unpack and reverse sort
        _, unsort_idx = sort_idx.sort(0)
        output1 = output1[unsort_idx]
        output2 = output2[unsort_idx]

        return output1, output2


# Attention layer
class FullAttention(nn.Module):
    def __init__(self, args, full_size, hidden_size, num_level):
        super(FullAttention, self).__init__()
        assert (hidden_size % num_level == 0)
        self.full_size = full_size
        self.hidden_size = hidden_size
        self.attsize_per_lvl = hidden_size // num_level
        self.num_level = num_level
        self.linear = nn.Linear(full_size, hidden_size, bias=False)
        self.linear_final = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.output_size = hidden_size

        self.args = args

        print("Full Attention: (atten. {} -> {}, take {}) x {}".format(self.full_size, self.attsize_per_lvl,
                                                                       hidden_size // num_level, self.num_level))

    def forward(self, x1_att, x2_att, x2, x2_mask):
        """
        x1_att: batch * len1 * full_size
        x2_att: batch * len2 * full_size
        x2: batch * len2 * hidden_size
        x2_mask: batch * len2
        """

        len1 = x1_att.size(1)
        len2 = x2_att.size(1)

        x1_att = F.dropout(x1_att, p=self.args.drop_prob, training=self.training)
        x2_att = F.dropout(x2_att, p=self.args.drop_prob, training=self.training)

        x1_key = F.relu(self.linear(x1_att.view(-1, self.full_size)))
        x2_key = F.relu(self.linear(x2_att.view(-1, self.full_size)))
        final_v = self.linear_final.expand_as(x2_key)
        x2_key = final_v * x2_key

        x1_rep = x1_key.view(-1, len1, self.num_level, self.attsize_per_lvl).transpose(1, 2).contiguous().view(-1, len1,
                                                                                                               self.attsize_per_lvl)
        x2_rep = x2_key.view(-1, len2, self.num_level, self.attsize_per_lvl).transpose(1, 2).contiguous().view(-1, len2,
                                                                                                               self.attsize_per_lvl)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2)).view(-1, self.num_level, len1,
                                                         len2)  # batch * num_level * len1 * len2

        x2_mask = x2_mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
        scores.data.masked_fill_(x2_mask.data, -1e30)

        alpha_flat = F.softmax(scores.view(-1, len2), dim=-1)
        alpha = alpha_flat.view(-1, len1, len2)

        size_per_level = self.hidden_size // self.num_level
        atten_seq = alpha.bmm(
            x2.contiguous().view(-1, len2, self.num_level, size_per_level).transpose(1, 2).contiguous().view(-1, len2,
                                                                                                             size_per_level))

        return atten_seq.view(-1, self.num_level, len1, size_per_level).transpose(1, 2).contiguous().view(-1, len1,
                                                                                                          self.hidden_size)


# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """
    Self attention over a sequence:
    """

    def __init__(self, args, input_size, hidden_size=1, is_output=False):
        super(LinearSelfAttn, self).__init__()

        self.args = args

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_output = is_output

        self.linear = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x1, x2, x2_mask):
        """

        """

        softmax_fn = F.log_softmax if self.is_output else F.softmax

        if x1 is None:

            x2 = F.dropout(x2, p=self.args.drop_prob, training=self.training)

            x2_flat = x2.contiguous().view(-1, x2.size(-1))
            scores = self.linear(x2_flat).view(x2.size(0), x2.size(1))
            scores.data.masked_fill_(x2_mask.data, -1e30)

            alpha = softmax_fn(scores, dim=-1)

        else:

            len2 = x2.size(1)

            x1 = F.dropout(x1, p=self.args.drop_prob, training=self.training)
            x2 = F.dropout(x2, p=self.args.drop_prob, training=self.training)

            x2_flat = x2.contiguous().view(-1, self.input_size)
            x2_key = self.linear(x2_flat).view(-1, len2, self.hidden_size)

            x1_rep = x1.unsqueeze(-1).transpose(1, 2)

            x2_rep = x2_key
            scores = x1_rep.bmm(x2_rep.transpose(1, 2)).squeeze(1)

            scores.data.masked_fill_(x2_mask.data, -1e30)
            alpha_flat = softmax_fn(scores, dim=-1)

            alpha = alpha_flat

        result = alpha

        if self.is_output == False:
            atten_seq = alpha.unsqueeze(-1).transpose(1, 2).bmm(x2).transpose(1, 2).squeeze(-1)
            result = atten_seq

        return result
