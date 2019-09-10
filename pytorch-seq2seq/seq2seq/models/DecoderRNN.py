import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False, use_concept=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, dialog_hidden=128, embedding=128):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        if use_concept:
            self.rnn = self.rnn_cell(hidden_size * 2 + embedding, hidden_size, n_layers, batch_first=True,
                                     dropout=dropout_p)
        else:
            self.rnn = self.rnn_cell(hidden_size * 2, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.hidden_size = hidden_size

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.copy_c = nn.Linear(dialog_hidden, 64)
        self.copy_e = nn.Linear(embedding, 64)
        self.copy_o = nn.Linear(embedding, 64)
        self.copy_match = nn.Linear(192, 64)
        self.copy_h = nn.Linear(hidden_size, 64)
        self.copy_distribution = nn.Linear(64, 1)

        self.choose_gate = nn.Linear(self.hidden_size * 2, 1)

    def forward_step(self, input_var, hidden, encoder_outputs, function, mix=None, score_copy=None, use_copy=False,
                     concept_rep=None):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        """
        先用上一轮预测的word更新decoder状态，然后计算p_c和p_g的权衡choose gate
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        tmp = torch.zeros_like(concept_rep)
        if use_copy:
            decoder_input = torch.cat([mix, embedded, tmp], dim=-1)
        else:
            decoder_input = torch.cat([mix, embedded], dim=-1)
        output, hidden = self.rnn(decoder_input, hidden)

        choose_rate = torch.sigmoid(self.choose_gate(torch.cat([hidden.squeeze(), embedded.squeeze()], dim=1)))

        # 对所有input做attention，mix就是所有encoder hidden state的加权和
        attn = None
        if self.use_attention:
            output, attn, mix = self.attention(output, encoder_outputs)
        # 算出p_g
        score_vocab = self.out(output.contiguous().view(-1, self.hidden_size))
        mean = score_vocab.mean(dim=1).unsqueeze(1)
        std = score_vocab.std(dim=1).unsqueeze(1)
        score_vocab = (score_vocab - mean) / std
        score_vocab = torch.softmax(score_vocab, dim=1)

        # 使用choose gate对p_g和p_c做加权和
        if use_copy:
            final_score = score_vocab * choose_rate + score_copy * (1 - choose_rate)
        else:
            final_score = score_vocab
        predicted_softmax = final_score

        # 这个要做log，是传出去计算loss的
        predicted_softmax = torch.log(predicted_softmax)
        return predicted_softmax, hidden, attn, mix

    # 这个函数用来计算p_c（copy概率分布），基本思想是将各concept的embedding和当前信息匹配，算一个概率分布
    # res_tmp就是当前信息的总结，res_e就是embedding经过线性变换的总结，两者做内积再经过标准化、softmax得到
    # 最终的分布copy_distribution
    # 此外，将各concept的embedding按照概率分布做加权和，得到一个总结性的向量concept_summary，论文里记号为o
    def copy(self, context, decoder_state, dialog_state, concepts, embeddings, tgt_vocab, cpt_vocab, concept_rep):

        res_c = self.copy_c(context)
        res_h = self.copy_h(decoder_state)
        res_o = self.copy_o(concept_rep.squeeze())
        tmp = torch.cat([res_c, res_h, res_o], dim=-1)
        res_tmp = self.copy_match(tmp).unsqueeze(2)
        res_e = self.copy_e(embeddings)
        score = torch.bmm(res_e, res_tmp).reshape(len(decoder_state), -1)
        # score = self.copy_distribution(res_c + res_e + res_h).reshape(len(concepts), -1)
        mean = score.mean(dim=1).unsqueeze(1)
        std = score.std(dim=1).unsqueeze(1)
        score = (score - mean) / std

        # copy_distribution = torch.exp(score)
        copy_distribution = torch.softmax(score, dim=1)
        concept_summary = torch.bmm(copy_distribution.unsqueeze(1), embeddings)

        # 下面所做的事是把concept_distribution从状态集映射成vocabulary上的概率分布，最终得到
        # score_copy就是 batch * vocab 长度的向量
        mapped_concepts = []
        max_len = max([len(line) for line in concepts])
        for i in range(len(concepts)):
            mapped_sent = []
            for j in range(len(concepts[i])):
                cpt = cpt_vocab.itos[concepts[i][j]]
                if cpt in tgt_vocab.stoi:
                    mapped_sent.append(tgt_vocab.stoi[cpt])
                else:
                    mapped_sent.append(self.output_size)
            mapped_sent.extend((max_len - len(concepts[i])) * [self.output_size])
            mapped_concepts.append(mapped_sent)
        mapped_concepts_tensor = torch.tensor(mapped_concepts)
        score_copy = torch.zeros((len(concepts), self.output_size + 1))
        if torch.cuda.is_available():
            mapped_concepts_tensor = mapped_concepts_tensor.cuda()
            score_copy = score_copy.cuda()
        score_copy = score_copy.scatter(1, mapped_concepts_tensor, copy_distribution)
        score_copy = score_copy[:, :-1]
        if torch.cuda.is_available():
            score_copy = score_copy.cuda()
        """
        # batch * hidden/dialog
        res_c = self.copy_c(context)
        res_h = self.copy_h(decoder_state)
        score_copy = torch.zeros([len(context), self.output_size])
        if torch.cuda.is_available():
            score_copy = score_copy.cuda()
        # batch * num_concepts * embedding
        for i in range(len(concepts)):
            res_e = self.copy_e(embeddings[i])
            score = self.copy_distribution(res_c[i] + res_e + res_h[i])
            final_score = torch.zeros(self.output_size)
            if torch.cuda.is_available():
                final_score = final_score.cuda()
            for j, cpt in enumerate(concepts[i]):
                score[j] *= dialog_state[i][cpt]
            score = score.reshape(-1)
            score = torch.exp(score)
            for j, cpt in enumerate(concepts[i]):
                string_form = cpt_vocab.itos[cpt]
                if string_form in tgt_vocab.stoi:
                    final_score[tgt_vocab.stoi[string_form]] += score[j]
            score_copy[i] += final_score
        """
        return score_copy, concept_summary

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax,
                teacher_forcing_ratio=0, batch_state=None, batch_concepts=None, batch_embeddings=None, context=None,
                cpt_vocab=None, tgt_vocab=None, use_copy=False, concept_rep=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # 这个就是把encoder传过来的数据稍微reshape一下以适合decoder
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        # 使用encoder_hidden来初始化decoder
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        # 这个函数就是个数据处理函数，是原作者写的，大概就是把decoder传回来的信息总结总结放在一个结构体里
        # 可以不用管
        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            # decoder_input = inputs[:, :-1]
            for di in range(max_length):
                decoder_input = inputs[:, di].unsqueeze(1)
                if di == 0:
                    mix = torch.zeros(batch_size, 1, self.hidden_size)
                    if torch.cuda.is_available():
                        mix = mix.cuda()
                if use_copy:
                    # 下面这两个必须注释其中一个，要么使用copy，要么不用copy。下面那个是用来做
                    # ablation test的
                    """
                    score_copy, concept_summary = self.copy(context, decoder_hidden.squeeze(), batch_state,
                                                            batch_concepts,
                                                            batch_embeddings, tgt_vocab, cpt_vocab, concept_rep)
                                                            """
                    score_copy = torch.zeros([len(context), self.output_size])

                    if torch.cuda.is_available():
                        score_copy = score_copy.cuda()
                    decoder_output, decoder_hidden, step_attn, mix = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs, mix=mix,
                                                                                       function=function,
                                                                                       score_copy=score_copy,
                                                                                       use_copy=use_copy,
                                                                                       concept_rep=concept_rep)
                else:
                    decoder_output, decoder_hidden, step_attn, mix = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs,
                                                                                       mix=mix,
                                                                                       function=function)
                step_output = decoder_output.squeeze(1)
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                if di == 0:
                    mix = torch.zeros(batch_size, 1, self.hidden_size)
                if torch.cuda.is_available():
                    mix = mix.cuda()
                if use_copy:
                    """
                    score_copy, concept_summary = self.copy(context, decoder_hidden.squeeze(), batch_state,
                                                            batch_concepts,
                                                            batch_embeddings, tgt_vocab, cpt_vocab, concept_rep)
                    """
                    score_copy = torch.zeros([len(context), self.output_size])
                    if torch.cuda.is_available():
                        score_copy = score_copy.cuda()
                    decoder_output, decoder_hidden, step_attn, mix = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs,
                                                                                       function=function,
                                                                                       score_copy=score_copy,
                                                                                       use_copy=use_copy,
                                                                                       mix=mix,
                                                                                       concept_rep=concept_rep)
                else:
                    decoder_output, decoder_hidden, step_attn, mix = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs,
                                                                                       function=function,
                                                                                       mix=mix)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
