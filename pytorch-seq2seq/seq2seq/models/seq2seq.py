import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy
import seq2seq
from seq2seq.util.print_state import print_state
from seq2seq.util.conceptnet_util import ConceptNet


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, dialog_encoder=None, decode_function=F.log_softmax, cpt_vocab=None,
                 hidden_size=128, concept_level='simple', mid_size=64, dialog_hidden=128, conceptnet_file=None):
        super(Seq2seq, self).__init__()
        self.concept_level = concept_level
        #if conceptnet_file:
        #    self.concept_net = ConceptNet(conceptnet_file)
        self.encoder = encoder
        self.decoder = decoder
        self.dialog_encoder = dialog_encoder
        self.decode_function = decode_function
        self.cpt_vocab = cpt_vocab
        if self.cpt_vocab:
            self.cpt_embedding = nn.Embedding(len(cpt_vocab.itos), hidden_size)
        self.layer_u = torch.nn.Linear(hidden_size * 2, mid_size)
        self.layer_c = torch.nn.Linear(dialog_hidden, mid_size)
        self.layer_e = torch.nn.Linear(hidden_size, mid_size)
        self.layer_att = torch.nn.Linear(mid_size, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.forget_u = torch.nn.Linear(hidden_size * 2, mid_size, bias=False)
        self.forget_c = torch.nn.Linear(dialog_hidden, mid_size, bias=False)
        self.forget_o = torch.nn.Linear(hidden_size, mid_size, bias=False)
        self.forget = torch.nn.Linear(mid_size, 1, bias=False)
        self.decoder_input_MLP = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden = hidden_size

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def extract_per_utt(self, input_variable, encoder_outputs, eou_index):
        input_index = input_variable.numpy() if not torch.cuda.is_available() else input_variable.cpu().numpy()
        eou_pos = [np.where(line == eou_index)[0] for line in input_index]
        utt_hidden = [torch.cat([encoder_outputs[j][i].unsqueeze(0) for i in eou_pos[j]], 0) for j in
                      range(input_variable.shape[0])]
        max_num_utt = max([len(line) for line in utt_hidden])
        for i in range(input_variable.shape[0]):
            if torch.cuda.is_available():
                utt_hidden[i] = torch.cat(
                    [utt_hidden[i], torch.zeros([max_num_utt - len(utt_hidden[i]), len(utt_hidden[0][0])]).cuda()])
            else:
                utt_hidden[i] = torch.cat(
                    [utt_hidden[i], torch.zeros([max_num_utt - len(utt_hidden[i]), len(utt_hidden[0][0])])])
        utt_hidden = [line.unsqueeze(0) for line in utt_hidden]
        return torch.cat(utt_hidden, 0), eou_pos

    def utterance_extract(self, input_variable, eou_index, pad_index):
        input_index = input_variable.numpy() if not torch.cuda.is_available() else input_variable.cpu().numpy()
        eou_pos = [np.where(line == eou_index)[0] for line in input_index]
        input_index = [[int(num) for num in line] for line in input_index]
        batch_size = len(eou_pos)
        utterance_batch = []
        for i in range(batch_size):
            utterances_sample = []
            last_pos = 0
            for index in list(eou_pos[i]):
                utterances_sample.append(input_index[i][last_pos:index])
                last_pos = index + 1
            utterance_batch.append(utterances_sample)
        valid_length = [[len(line) for line in sample] for sample in utterance_batch]
        max_sentence = max([len(line) for line in utterance_batch])
        for i in range(batch_size):
            if len(utterance_batch[i]) < max_sentence:
                for k in range(max_sentence - len(utterance_batch[i])):
                    utterance_batch[i].append([pad_index])
        for i in range(max_sentence):
            max_token = 0
            for j in range(batch_size):
                if len(utterance_batch[j][i]) > max_token:
                    max_token = len(utterance_batch[j][i])
            for j in range(batch_size):
                if len(utterance_batch[j][i]) < max_token:
                    utterance_batch[j][i].extend((max_token - len(utterance_batch[j][i])) * [pad_index])
        input_by_utterance = []
        for i in range(max_sentence):
            input_utterance = []
            for j in range(batch_size):
                input_utterance.append(utterance_batch[j][i])
            input_by_utterance.append(torch.tensor(input_utterance))
        if torch.cuda.is_available():
            input_by_utterance = [tensor.cuda() for tensor in input_by_utterance]
        return input_by_utterance, valid_length

    # input: a list of tensors with shape [*, embedding_size]
    def embedding_padding(self, embedding):
        max_len = max([len(line) for line in embedding])
        vector_length = embedding[0].shape[-1]
        zero = torch.zeros((1, vector_length))
        if torch.cuda.is_available():
            zero = zero.cuda()
        for i in range(len(embedding)):
            if len(embedding[i]) < max_len:
                embedding[i] = torch.cat([embedding[i], torch.cat((max_len - len(embedding[i])) * [zero])]).unsqueeze(0)
            else:
                embedding[i] = embedding[i].unsqueeze(0)
        return torch.cat(embedding)

    # return size: batch * num_sentence * num_concept_per_sentence * embedding
    def concept_mapping(self, concept, vocab, src_vocab):
        pad_index = vocab.stoi['<pad>']
        eou_index = vocab.stoi['<eou>']
        expand_index = vocab.stoi['<expand>']
        index_index = vocab.stoi['<index>']
        response_index = vocab.stoi['<response>']
        np_concept = concept.numpy() if not torch.cuda.is_available() else concept.cpu().numpy()
        end_pos = []
        for line in np_concept:
            pos = np.where(line == pad_index)[0]
            if len(pos):
                end_pos.append(pos[0])
            else:
                end_pos.append(len(line))
        np_concept = [np_concept[i][:end_pos[i]] for i in range(len(np_concept))]

        expanded_batch = []
        #indexes_batch = []
        concepts_batch = []
        response_concept = []
        for line in np_concept:
            concepts_batch.append(line[:np.where(line == expand_index)[0][0]])
            #expanded_batch.append(line[np.where(line == expand_index)[0][0]+1:np.where(line == index_index)[0][0]])
            expanded_batch.append(line[np.where(line == expand_index)[0][0]+1:np.where(line == response_index)[0][0]])
            #indexes_batch.append(line[np.where(line == index_index)[0][0]+1:])
            response_concept.append(line[np.where(line == response_index)[0][0]+1:])
        #indexes_batch = [[vocab.itos[word] for word in line] for line in indexes_batch]


        # newly added: we use source vocabulary rather than concept vocabulary
        concepts_test = [[vocab.itos[word] for word in line] for line in concepts_batch]
        expanded_test = [[vocab.itos[word] for word in line] for line in expanded_batch]
        response_test = [[vocab.itos[word] for word in line] for line in response_concept]
        concepts_batch_for_embedding = [[src_vocab.stoi[word] for word in line] for line in concepts_test]
        expanded_batch_for_embedding = [[src_vocab.stoi[word] for word in line] for line in expanded_test]
        response_batch_for_embedding = [[src_vocab.stoi[word] for word in line] for line in response_test]

        concept_batch = []
        embedding_batch = []
        embedding_expand = []
        embedding_response = []
        for i in range(len(concepts_batch)):
            concept_d = []
            concept_d_for_embedding = []
            utt_pos = np.where(concepts_batch[i] == eou_index)[0]
            utt_pos = np.concatenate([[-1], utt_pos])
            for j in range(1, len(utt_pos)):
                concept_d.append(concepts_batch[i][utt_pos[j - 1] + 1:utt_pos[j]])
                concept_d_for_embedding.append(concepts_batch_for_embedding[i][utt_pos[j - 1] + 1:utt_pos[j]])
            if torch.cuda.is_available():
                embedding_expand.append(self.encoder.embedding(torch.tensor(expanded_batch_for_embedding[i]).cuda()))
                embedding_response.append(self.encoder.embedding(torch.LongTensor(response_batch_for_embedding[i]).cuda()))
                concept_mapped = [self.encoder.embedding(torch.LongTensor(line).cuda()) for line in concept_d_for_embedding]
            else:
                embedding_expand.append(self.encoder.embedding(torch.tensor(expanded_batch_for_embedding[i])))
                embedding_response.append(self.encoder.embedding(torch.LongTensor(response_batch_for_embedding[i])))
                concept_mapped = [self.encoder.embedding(torch.LongTensor(line)) for line in concept_d_for_embedding]
            concept_batch.append(concept_d)
            embedding_batch.append(concept_mapped)
        embedding_expand = self.embedding_padding(embedding_expand)
        embedding_response = self.embedding_padding(embedding_response)
        #return concept_batch, embedding_batch, expanded_batch, indexes_batch, embedding_expand
        return concept_batch, embedding_batch, expanded_batch, embedding_expand, response_concept, embedding_response

    def state_track(self, concept, embedding, dialog, utterance, pad_index, expand_concept, expand_embed, cpt_vocab=None):
        max_sentence = max([len(line) for line in embedding])
        one = torch.ones((1, self.hidden))
        zero = torch.zeros((1, self.hidden))
        batch_size = len(concept)
        g = torch.ones([batch_size, 1])
        if torch.cuda.is_available():
            one = one.cuda()
            g = g.cuda()
            zero = zero.cuda()
        concept = [[list(line) for line in sample] for sample in concept]


        # batch padding
        for j in range(batch_size):
            if len(embedding[j]) < max_sentence:
                embedding[j].extend((max_sentence - len(embedding[j])) * [one])
                for k in range(max_sentence - len(concept[j])):
                    concept[j].append([pad_index])
        for i in range(max_sentence):
            max_concepts = 0
            for j in range(batch_size):
                num = embedding[j][i].shape[0]
                max_concepts = max_concepts if max_concepts >= num else num
            for j in range(batch_size):
                num = embedding[j][i].shape[0]
                if num < max_concepts:
                    embedding[j][i] = torch.cat([embedding[j][i], torch.cat([one] * (max_concepts - num))])
                    concept[j][i].extend((max_concepts - num) * [pad_index])

        # debug: get the first sample to track state
        debug_concept = [[cpt_vocab.itos[word] for word in line] for line in concept[0]]

        embedding_per_step = []
        for i in range(max_sentence):
            emb = []
            for j in range(batch_size):
                emb.append(embedding[j][i].unsqueeze(0))
            embedding_per_step.append(torch.cat(emb, 0))
        # calculating state
        for i in range(max_sentence):
            c = dialog[:, i-1] if i != 0 else torch.zeros_like(dialog[:, 0]) # bs * hidden
            u = utterance[:, i] # bs * (hidden*2)
            cpt = embedding_per_step[i]
            res_u = self.layer_u(u).unsqueeze(1)
            res_c = self.layer_c(c).unsqueeze(1)
            res_e = self.layer_e(cpt)
            distribution = self.softmax(self.layer_att(res_u + res_c + res_e).reshape(batch_size, -1))

            #o = torch.bmm(distribution.unsqueeze(1), cpt).squeeze()
            #res_f_u = self.forget_u(u)
            #res_f_c = self.forget_c(c)
            #res_f_o = self.forget_o(o)
            if i != 0:
                #g = self.sigmoid(self.forget(res_f_c + res_f_u))
                g = 0.8
                state = torch.cat([state * g, distribution * (1 - g)], 1)

                # debug
                """
                debug_distribution = distribution[0].detach().numpy()
                for j in range(len(debug_distribution)):
                    print(debug_concept[i][j], debug_distribution[j], end=' ')
                print('\nRemember Rate:', 0.8)
                """

            else:
                state = distribution

        # language planning
        #e = torch.cat([line.unsqueeze(0) for line in expand_embed])
        e = expand_embed
        res_c = self.layer_c(c).unsqueeze(1)
        res_u = self.layer_u(u).unsqueeze(1)
        res_e = self.layer_e(e)
        distribution = self.softmax(torch.tanh(self.layer_att(res_u + res_c + res_e)).reshape((batch_size, -1)))
        o = torch.bmm(distribution.unsqueeze(1), e).squeeze()
        res_f_c = self.forget_c(c)
        res_f_u = self.forget_u(u)
        res_f_o = self.forget_o(o)
        g = self.sigmoid(self.forget(res_f_c + res_f_u + res_f_o))
        state = torch.cat([state * g, distribution * (1 - g)], 1)

        concept_linear = []
        embedding_linear = torch.cat([torch.cat(line).unsqueeze(0) for line in embedding])
        embedding_linear = torch.cat((embedding_linear, expand_embed), dim=1)
        for i in range(batch_size):
            concepts = []
            for j in range(len(concept[i])):
                concepts.extend(concept[i][j])
            concepts.extend(expand_concept[i])
            concept_linear.append(concepts)

        """
        # filtered state template
        concept_linear = []
        embedding_linear = []
        dict_linear = []
        for k in range(batch_size):
            i_to_concept = []
            i_to_embedding = []
            concept_to_i = {}
            index = 0
            for i in range(len(concept[k])):
                if not len(concept[i]):
                    continue
                for cnt, cpt in enumerate(concept[k][i]):
                    if cpt not in i_to_concept:
                        i_to_concept.append(cpt)
                        i_to_embedding.append(embedding[k][i][cnt].unsqueeze(0))
                        concept_to_i[cpt] = index
                        index += 1

            for cnt, cpt in enumerate(list(expand_concept[k])):
                if cpt not in i_to_concept:
                    i_to_concept.append(cpt)
                    i_to_embedding.append(expand_embed[k][cnt].unsqueeze(0))
                    concept_to_i[cpt] = index
                    index += 1

            i_to_embedding = torch.cat(i_to_embedding, 0)

            concept_linear.append(i_to_concept)
            embedding_linear.append(i_to_embedding)
            dict_linear.append(concept_to_i)
        max_concepts = max([len(line) for line in concept_linear])
        states = torch.zeros((batch_size, max_concepts))
        if torch.cuda.is_available():
            states = states.cuda()
        for i in range(batch_size):
            if len(embedding_linear[i]) < max_concepts:
                tmp = torch.cat(((max_concepts - len(embedding_linear[i])) * [zero]))
                embedding_linear[i] = torch.cat((embedding_linear[i], tmp)).unsqueeze(0)
            else:
                embedding_linear[i] = embedding_linear[i].unsqueeze((0))
        embedding_linear = torch.cat(embedding_linear)

        # generate final state
        for i in range(batch_size):
            cnt = 0
            cpt_dict = dict_linear[i]
            for j in range(len(concept[i])):
                for k, cpt in enumerate(concept[i][j]):
                    if cpt != '<pad>':
                        states[i][cpt_dict[cpt]] += state[i][cnt]
                    cnt += 1
            for j, cpt in enumerate(expand_concept[i]):
                states[i][cpt_dict[cpt]] += state[i][cnt + j]


        concept_rep = []
        for i in range(batch_size):
            concept_rep.append(torch.mm(states[i].unsqueeze(0), embedding_linear[i]))
        """
        concept_rep = torch.bmm(state.unsqueeze(1), embedding_linear).squeeze()
        #concept_rep = torch.zeros((len(embedding), self.hidden))
        if torch.cuda.is_available():
            concept_rep = concept_rep.cuda()
        return state, concept_linear, embedding_linear, concept_rep

    def single_turn_state_track(self, concept_batch, embedding_batch, dialog, expand_batch, expand_embedding):
        concept_linear = []
        emb_linear = []
        zero = torch.zeros((1, self.hidden))
        if torch.cuda.is_available():
            zero = zero.cuda()
        for i in range(len(concept_batch)):
            res = []
            emb = []
            for j in range(len(concept_batch[i])):
                res.extend(list(concept_batch[i][j]))
                emb.extend(embedding_batch[i][j].unsqueeze(0))
            if len(emb) != 0:
                emb = torch.cat(emb, 0)
            else:
                emb = zero
            concept_linear.append(res)
            emb_linear.append(emb)

        for i in range(len(concept_batch)):
            emb_linear[i] = torch.cat([emb_linear[i], expand_embedding[i]], dim=0)
            concept_linear[i].extend(expand_batch[i])

        emb_linear = self.embedding_padding(emb_linear)

        c = dialog[:, -1]
        res_c = self.layer_c(c)
        res_c = res_c.reshape(res_c.shape[0], 1, res_c.shape[-1])
        res_e = self.layer_e(emb_linear)
        res = self.layer_att(res_e + res_c).squeeze()
        res = self.softmax(res)

        # o是所有concept的embedding加权和，相当于concept的总结向量
        #o = torch.bmm(res.unsqueeze(1), emb_linear).squeeze()
        o = torch.zeros((len(emb_linear), self.hidden)).cuda()
        return res, concept_linear, emb_linear, o

    def extend_batch(self, batch_concepts, cpt_vocab, batch_states):
        new_concepts = [[cpt_vocab.itos[word] for word in line] for line in batch_concepts]
        index = [self.concept_net.expand_list(line)[1] for line in new_concepts]
        new_concepts = [self.concept_net.expand_list(line)[0] for line in new_concepts]
        new_concepts = [[word for word in line if word in cpt_vocab.stoi] for line in new_concepts]
        max_concepts = max([len(line) for line in new_concepts])
        for i in range(len(new_concepts)):
            if len(new_concepts[i]) < max_concepts:
                new_concepts[i].extend((max_concepts - len(new_concepts[i])) * ['<pad>'])
        new_concepts = [[cpt_vocab.stoi[word] for word in line] for line in new_concepts]
        embeddings = self.cpt_embedding(torch.tensor(new_concepts))
        print(sum([len(line) for line in new_concepts]) / len(new_concepts))

    def language_planning(self, batch_states, origin_concepts, expand_batch, index_batch, vocab, context, last_utt):
        max_concepts = max([len(line) for line in expand_batch])
        expand_batch = [list(line) for line in expand_batch]
        for i in range(len(expand_batch)):
            if len(expand_batch[i]) < max_concepts:
                expand_batch[i].extend((max_concepts - len(expand_batch[i])) * [vocab.stoi['<pad>']])
        embedding = self.cpt_embedding(torch.tensor(expand_batch))
        res_e = self.layer_e(embedding)
        res_u = self.layer_u(last_utt).unsqueeze(1)
        res_c = self.layer_c(context)
        res = self.softmax(torch.nn.functional.tanh(self.layer_att(res_e + res_u + res_c)).reshape(len(batch_states), -1))

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, concept=None, vocabs=None, use_concept=False, track_state=False):
        if use_concept:
            src_vocab = vocabs.src_vocab
            tgt_vocab = vocabs.tgt_vocab
            cpt_vocab = vocabs.cpt_vocab
            eou_index = src_vocab.stoi['<eou>']
            pad_index = src_vocab.stoi['<pad>']

            # encode the dialog utterance by utterance
            input_by_utterance, valid_length = self.utterance_extract(input_variable, eou_index, pad_index)
            encoder_hidden_dialog = []
            all_encoder_hiddens = []
            encoder_hidden = []
            for k in range(len(input_by_utterance)):
                if len(encoder_hidden):
                    encoder_outputs, encoder_hidden = self.encoder(input_by_utterance[k], encoder_hidden[0])
                else:
                    encoder_outputs, encoder_hidden = self.encoder(input_by_utterance[k])
                encoder_hidden_dialog.append(torch.cat((encoder_hidden[0], encoder_hidden[1]), 1).unsqueeze(1))
                for i in range(len(valid_length)):
                    if len(valid_length[i]) < k + 1:
                        continue
                    if len(all_encoder_hiddens) < i+1:
                        all_encoder_hiddens.append([encoder_outputs[i, :valid_length[i][k]]])
                    else:
                        all_encoder_hiddens[i].append(encoder_outputs[i, :valid_length[i][k]])
            encoder_hidden_dialog = torch.cat(encoder_hidden_dialog, dim=1)
            all_encoder_hiddens = [torch.cat(sample) for sample in all_encoder_hiddens]
            zero = torch.zeros((1, all_encoder_hiddens[0].shape[-1]))
            if torch.cuda.is_available():
                zero = zero.cuda()
            max_len = max([len(line) for line in all_encoder_hiddens])
            for k in range(len(all_encoder_hiddens)):
                if len(all_encoder_hiddens[k]) < max_len:
                    all_encoder_hiddens[k] = torch.cat((all_encoder_hiddens[k], torch.cat((max_len - len(all_encoder_hiddens[k])) * [zero])))
            all_encoder_hiddens = [line.unsqueeze(0) for line in all_encoder_hiddens]
            all_encoder_hiddens = torch.cat(all_encoder_hiddens, 0)

            dialog_output, (context, _) = self.dialog_encoder(encoder_hidden_dialog)
            concept_batch, embedding_batch, expanded_batch, embedding_expand, response_concept, embedding_response = self.concept_mapping(concept, cpt_vocab, src_vocab)
            batch_state = []
            batch_concepts = []
            batch_embeddings = []
            if self.concept_level == 'simple':
                batch_state, batch_concepts, batch_embeddings, o = self.single_turn_state_track(concept_batch,
                                                                                                embedding_batch,
                                                                                                dialog_output,
                                                                                                expanded_batch,
                                                                                                embedding_expand)
            elif self.concept_level == 'complex':
                batch_state, batch_concepts, batch_embeddings, o = self.state_track(concept_batch, embedding_batch,
                                                                                    dialog_output, encoder_hidden_dialog,
                                                                                    pad_index, expanded_batch, embedding_expand, cpt_vocab)
            if track_state:
                res_all = []
                for i in range(len(batch_concepts)):
                    res = print_state(batch_concepts[i], batch_state[i], cpt_vocab)
                    res_all.append(res)

            # handling response concepts
            mapped_concepts = []
            max_cpt_len = max([len(line) for line in batch_concepts])
            for i in range(len(batch_concepts)):
                if len(batch_concepts[i]) < max_cpt_len:
                    mapped_concepts.append(batch_concepts[i] + (max_cpt_len - len(batch_concepts[i])) * [cpt_vocab.stoi['<pad>']])
                else:
                    mapped_concepts.append(batch_concepts[i])
            mapped_concepts = torch.tensor(mapped_concepts)
            state_score = torch.zeros((len(batch_state), len(cpt_vocab.itos)))
            if torch.cuda.is_available():
                mapped_concepts = mapped_concepts.cuda()
                state_score = state_score.cuda()
            state_score = state_score.scatter(1, mapped_concepts, batch_state)
            response_concept = [[word for word in line if word in batch_concepts[k] and word != cpt_vocab.stoi['<unk>']] for k, line in enumerate(response_concept)]
            """
            max_len = max([len(line) for line in response_concept])
            for i in range(len(response_concept)):
                if len(response_concept[i]) < max_len:
                    response_concept[i].extend((max_len - len(response_concept[i])) * [cpt_vocab.stoi['<pad>']])
            response_concept = torch.tensor(response_concept)
            """

            last_state = dialog_output[:, -1, :].unsqueeze(0)
            #last_state = torch.cat((last_state, o.unsqueeze(0)))
            last_state = torch.cat((last_state, torch.zeros_like(last_state)))
            o = o.unsqueeze(1)
            #last_state = torch.cat((last_state, o), dim=-1)
            #decoder_input = self.decoder_input_MLP(last_state)
            result = self.decoder(inputs=target_variable,
                                  encoder_hidden=last_state,
                                  encoder_outputs=all_encoder_hiddens,
                                  function=self.decode_function,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  batch_state=batch_state,
                                  batch_concepts=batch_concepts,
                                  batch_embeddings=batch_embeddings,
                                  context=context.squeeze(),
                                  cpt_vocab=cpt_vocab,
                                  tgt_vocab=tgt_vocab,
                                  use_copy=use_concept,
                                  concept_rep=o)
        else:
            encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
            result = self.decoder(inputs=target_variable,
                                  encoder_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  function=self.decode_function,
                                  teacher_forcing_ratio=teacher_forcing_ratio)
        if track_state and use_concept:
            return result, (torch.log(state_score), response_concept), res_all
        elif use_concept:
            return result, (torch.log(state_score), response_concept)
        else:
            return result
