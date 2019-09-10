from __future__ import print_function, division

import torch
import torchtext

import seq2seq
import autoeval
from seq2seq.loss import NLLLoss
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from autoeval.eval_embedding import Embed
from autoeval.eval_distinct import distinct

smoothie = SmoothingFunction().method4


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data, vocabs=None, use_concept=False, log_dir=None, embed=None, cur_step=0):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """

        eval_limit = 5000
        step_limit = int(eval_limit / self.batch_size)

        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = torch.device('cuda', 0) if torch.cuda.is_available() else None
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        src_vocab = data.fields[seq2seq.src_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        cnt = 0
        loss_sum = 0

        context_corpus = []
        reference_corpus = []
        prediction_corpus = []
        state_corpus = []
        with torch.no_grad():
            for batch in batch_iterator:
                cnt += 1
                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)

                if torch.cuda.is_available():
                    input_index = input_variables.cpu().numpy()
                else:
                    input_index = input_variables.numpy()
                input_words = [[src_vocab.itos[word] for word in line] for line in input_index]
                context_corpus.extend(input_words)

                if use_concept:
                    concept, _ = getattr(batch, seq2seq.cpt_field_name)
                else:
                    concept = []
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                if use_concept:
                    (decoder_outputs, decoder_hidden, other), state_loss, state_print = model(input_variables,
                                                                                              input_lengths.tolist(),
                                                                                              target_variables,
                                                                                              concept=concept,
                                                                                              vocabs=vocabs,
                                                                                              use_concept=use_concept,
                                                                                              track_state=use_concept)
                    state_corpus.extend(state_print)
                    """
                    decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(),
                                                                            target_variables,
                                                                            concept=concept, vocabs=vocabs,
                                                                            use_concept=use_concept,
                                                                            track_state=False)
                    """
                else:
                    decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(),
                                                                   target_variables, vocabs=vocabs)
                # Evaluation
                seqlist = other['sequence']
                reference = []
                prediction = []
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)
                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()
                    if torch.cuda.is_available():
                        pred = seqlist[step].view(-1).cpu().numpy()
                        tgt = target.view(-1).cpu().numpy()
                    else:
                        pred = seqlist[step].view(-1).numpy()
                        tgt = target.view(-1).numpy()
                    for i in range(len(step_output)):
                        target_char = tgt_vocab.itos[tgt[i]]
                        pred_char = tgt_vocab.itos[pred[i]]
                        if target_char != '<pad>':
                            if len(reference) >= i + 1:
                                reference[i].append(target_char)
                            else:
                                reference.append([target_char])
                        if pred_char != '<pad>':
                            if len(prediction) >= i + 1:
                                if prediction[i][-1] != '<eos>':
                                    prediction[i].append(pred_char)
                            else:
                                prediction.append([pred_char])
                for i in range(len(reference)):
                    reference[i] = reference[i][:-1]
                    prediction[i] = prediction[i][:-1]
                reference_corpus.extend([[line] for line in reference])
                prediction_corpus.extend(prediction)
                if cnt > step_limit:
                    break

        bleu = corpus_bleu(reference_corpus, prediction_corpus, smoothing_function=smoothie)
        # embedding = embed.eval_embedding(reference_corpus, prediction_corpus)
        distinct_1 = distinct(prediction_corpus, 1)
        distinct_2 = distinct(prediction_corpus, 2)
        print("Corpus BLEU: ", bleu)
        # print("Embedding dist: ", embedding)
        print("Distinct-1: ", distinct_1)
        print("Distinct-2: ", distinct_2)
        with open(log_dir + '/log.txt', 'a+', encoding='utf-8') as file:
            file.write("Distinct-1: " + str(distinct_1) + '\n')
            file.write("Distinct-2: " + str(distinct_2) + '\n\n')

        with open(log_dir + '/log-' + str(cur_step), 'w', encoding='utf-8') as file:
            file.write("Corpus BLEU: " + str(bleu) + '\n')
            # file.write("Embedding Dist: " + str(embedding) + '\n')
            file.write("Distinct-1: " + str(distinct_1) + '\n')
            file.write("Distinct-2: " + str(distinct_2) + '\n\n')
            for i in range(len(reference_corpus)):
                file.write("Context: " + '\n')
                context_str = " ".join(context_corpus[i])
                context_list = context_str.split('<eou>')
                for j in range(len(context_list)):
                    file.write(context_list[j] + '\n')
                if use_concept and state_corpus:
                    file.write("\nStates: " + '\n')
                    cd_pairs = zip(state_corpus[i][0], state_corpus[i][1])
                    cd_pairs = sorted(set(cd_pairs), key=lambda x: x[1])
                    for j in range(len(state_corpus[i][0])):
                        file.write("Concept: {}. Prob: {}.\n".format(cd_pairs[j][0], cd_pairs[j][1]))
                file.write("\nGold: " + ' '.join(reference_corpus[i][0]) + '\n\n')
                file.write("Response: " + ' '.join(prediction_corpus[i]) + '\n\n')
                file.write('\n')
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
