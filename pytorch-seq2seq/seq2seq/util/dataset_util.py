import json
from nltk.tokenize import wordpunct_tokenize
from conceptnet_util import ConceptNet
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import copy
import pickle
import numpy as np
import time


def load_data(path):
    data = json.loads(open(path).read())
    dialogs = []
    docs_for_tfidf = []
    for line in data:
        dialog = line['dialog']
        utterances = []
        utt_linear = ""
        for utterance in dialog:
            text = utterance['text']
            tokenized = wordpunct_tokenize(text)
            utterances.append(tokenized)
            utt_linear += text + ' '
        dialogs.append(utterances)
        docs_for_tfidf.append(utt_linear)
    return dialogs, docs_for_tfidf


def get_tfidf_list(docs_for_tfidf):
    cv = CountVectorizer()
    wcv = cv.fit_transform(docs_for_tfidf)
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(wcv)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
    ascend_result = df_idf.sort_values(by=['idf_weights'])
    return ascend_result


def get_concepts(data, cn, stopwords, vocab):
    cpt_per_utt = [[list(set([word for word in sentence if word in cn.cpt_dict and word not in stopwords and word in vocab])) for sentence in dialog] for dialog in data]
    cpt_linear = []
    for dialog in cpt_per_utt:
        cpt = []
        for sent in dialog:
            cpt.extend(sent)
        cpt_linear.append(cpt)
    return cpt_per_utt, cpt_linear


def get_topK_by_tfidf(cpt_per_utt, cpt_linear, tf_idf, K):
    result = []
    dialogs_linear = []
    for i, dialog in enumerate(cpt_linear):
        in_dict = [word for word in dialog if word in tf_idf]
        res = sorted(in_dict, key=lambda x: tf_idf[x], reverse=True)
        if len(res) > K:
            result.append(res[:K])
        else:
            result.append(res)
        for j, utt in enumerate(cpt_per_utt[i]):
            cpt_per_utt[i][j] = [word for word in cpt_per_utt[i][j] if word in result[-1]]
        dialog_linear = []
        for utt in cpt_per_utt[i]:
            dialog_linear.extend(utt)
        dialogs_linear.append(dialog_linear)
    return dialogs_linear


def adjacent_ratio(response, context, cn):
    cnt = 0
    for i in range(len(context)):
        (res, _) = cn.expand_list(context[i])
        for cpt in response[i]:
            if cpt in res:
                cnt += 1
    return cnt / sum([len(line) for line in response])


def expand_by_path(concepts, k, cn, vocab=None, stopword=None):
    expanded = []
    total = len(concepts)
    per_step = int(total / 20)
    time_start = time.time()
    for i in range(len(concepts)):
        expanded.append(cn.expand_list_by_path(concepts[i], k, vocab, stopword))
        if per_step > 0 and i % per_step == 0:
            print("{}% completed.".format(i * 100 / total))
    end_time = time.time()
    print((end_time - time_start) / len(concepts))
    return expanded


def split_cpt(concept, k):
    cpt_ctx = []
    cpt_res = []
    for dialog in concept:
        ctx = []
        for i in range(len(dialog) - k):
            ctx.extend(dialog[i])
        cpt_ctx.append(ctx)
        cpt_res.append(dialog[len(dialog)-k])
    return cpt_ctx, cpt_res


def write_file(path, dialogs, concepts, expanded, indexes=[], cpt_res=None):
    with open(path, 'w') as f:
        for i in range(len(dialogs)):
            dialog_str = ""
            concept_str = ""
            for j in range(len(dialogs[i]) -1):
                dialog_str += " ".join(dialogs[i][j]) + " <eou> "
                concept_str += " ".join(concepts[i][j]) + " <eou> "
            concept_str += " <expand> " + " ".join(expanded[i])
            if cpt_res:
                concept_str += " <response> " + " ".join(cpt_res[i])
            response = " ".join(dialogs[i][-1])
            if indexes:
                index_str = " ".join([str(fig) for fig in indexes[i]])
                f.write(dialog_str + '\t' + response + '\t' + concept_str + " <index> " + index_str + '\n')
            else:
                f.write(dialog_str + '\t' + response + '\t' + concept_str + '\n')


def guide_rate_concept(cpt_res, cpt_ctx):
    count = 0
    tmp = [[word for word in cpt_res[i] if word in cpt_ctx[i]] for i in range(len(cpt_ctx))]
    return sum([len(line) for line in tmp]) / sum([len(line) for line in cpt_res])


# the ratio of responses that can be guided by the context
def guide_rate(cpt_res, cpt_ctx):
    count = 0
    for i in range(len(cpt_res)):
        for word in cpt_res[i]:
            if word in cpt_ctx[i]:
                count += 1
                break
    return count / len(cpt_res)


# the ratio of concepts that are related to the response, sorted by distance
def distance_rate(cpt_res, cpt_per_utt, cn):
    distance_dict = {}
    cpt_ctx = [line[:-1] for line in cpt_per_utt]
    for i in range(len(cpt_res)):
        candidates, _ = cn.expand_list(cpt_res[i])
        for j in range(len(cpt_ctx[i])):
            count = sum([1 for word in cpt_ctx[i][j] if word in candidates])
            pos = len(cpt_ctx[i]) - j
            if pos not in distance_dict:
                distance_dict[pos] = [count, len(cpt_ctx[i][j])]
            else:
                distance_dict[pos][0] += count
                distance_dict[pos][1] += len(cpt_ctx[i][j])
    return distance_dict


# the ratio of concepts in response that are related to the context, sorted by distance
def distance_rate_by_response(cpt_res, cpt_per_utt, cn):
    distance_dict = {}
    cpt_ctx = [line[:-1] for line in cpt_per_utt]
    for i in range(len(cpt_res)):
        for j in range(len(cpt_ctx[i])):
            candidates, _ = cn.expand_list(cpt_ctx[i][j])
            count = sum([1 for word in cpt_res[i] if word in candidates])
            pos = len(cpt_ctx[i]) - j
            if pos not in distance_dict:
                distance_dict[pos] = [count, len(cpt_res[i])]
            else:
                distance_dict[pos][0] += count
                distance_dict[pos][1] += len(cpt_res[i])
    return distance_dict


# the shortest distance of a response to be guided
def guide_distance(cpt_res, cpt_per_utt, cn):

    def guide(res, ctx, cn):
        for word in res:
            cnt = sum([1 for cpt in ctx if word in cn.cpt_dict[cpt]])
            if cnt > 0:
                return True
        return False

    distance_dict = {}
    for i in range(len(cpt_res)):
        for j in range(len(cpt_per_utt[i]) - 1):
            pos = len(cpt_per_utt[i]) - j - 2
            if guide(cpt_res[i], cpt_per_utt[i][pos], cn):
                if j + 1 in distance_dict:
                    distance_dict[j + 1] += 1
                else:
                    distance_dict[j + 1] = 1
                break
    return distance_dict


# decay detection
def decay_detection(cpt_res, cpt_per_utt, cn):
    distance_dict = {}

    def transfer(per_utt):
        new_concepts = copy.deepcopy(per_utt)
        for i in range(len(new_concepts)):
            for word in per_utt[i]:
                for j in reversed(range(i+1, len(per_utt))):
                    if sum([1 for cpt in per_utt[j] if word in cn.cpt_dict[cpt]]) or word in per_utt[j]:
                        new_concepts[i].remove(word)
                        if word not in new_concepts[j]:
                            new_concepts[j].append(word)
                        break
        return new_concepts
    for i in range(len(cpt_res)):
        new_ctx = transfer(cpt_per_utt[i][:-1])
        for j in range(len(new_ctx)):
            pos = len(new_ctx) - j
            for word in new_ctx[j]:
                if sum([1 for cpt in cpt_res[i] if cpt in cn.cpt_dict[word]]):
                    if pos not in distance_dict:
                        distance_dict[pos] = [1, 1]
                    else:
                        distance_dict[pos][0] += 1
                if pos in distance_dict:
                    distance_dict[pos][1] += 1
                else:
                    distance_dict[pos] = [0, 1]
    return distance_dict


def adjacent_num(cpt, cpt_list, cn):
    return len([word for word in cpt_list if cpt in cn.cpt_dict[word]])


def adjacent_avg(cpt_ctx, cpt_res, cn):
    ls = [adjacent_num(word, cpt_ctx, cn) for word in cpt_res]
    ls = [num for num in ls if num != 0]
    return sum(ls) / len(ls)


def adjacent_corpus(cpt_ctx, cpt_res, cn):
    time_dict = {}
    for i in range(len(cpt_ctx)):
        res = [adjacent_num(word, cpt_ctx[i], cn) for word in cpt_res[i]]
        res = [num for num in res if num != 0]
        for num in res:
            if num in time_dict:
                time_dict[num] += 1
            else:
                time_dict[num] = 1
    return time_dict


def adjacent_time(cpt_ctx, cpt_res, cn, vocab, stopword):
    time_dict = {}
    for i in range(len(cpt_ctx)):
        tmp_dict = cn.expand_list_by_path(cpt_ctx[i], 0, vocab, stopword, return_dict=True)
        for item in tmp_dict.items():
            if item[0] not in time_dict:
                time_dict[item[0]] = item[1]
            else:
                time_dict[item[0]] += item[1]
    return time_dict


def get_comparison(cpt_ctx, cpt_res, cn, vocab, stopwords):
    time_dict = adjacent_time(cpt_ctx, cpt_res, cn, vocab, stopwords)
    res_dict = adjacent_corpus(cpt_ctx, cpt_res, cn)
    total_time = sum(time_dict.values())
    total_res = sum(res_dict.values())
    for item in time_dict.items():
        time_dict[item[0]] /= total_time
    for item in res_dict.items():
        res_dict[item[0]] /= total_res
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.arange(1, 32, 1)
    v1 = [item[1] for item in sorted(time_dict.items(), key=lambda x: x[0])]
    v2 = [item[1] for item in sorted(res_dict.items(), key=lambda x: x[0])]
    ax.plot(x[:20], v1[:20])
    ax.plot(x[:20], v2[:20])
    fig.show()


def main():
    vocab = pickle.load(open("vocab", "rb"))
    vocab_full = pickle.load(open("vocab_full", "rb"))
    path = '../../../ConceptNet/valid'
    cn = ConceptNet("../../../ConceptNet/concept_dict.json")
    stopwords = [word.strip() for word in open('../../../ConceptNet/stopword.txt').readlines()]
    stopwords = {word: 1 for word in stopwords}
    data = pickle.load(open(path, "rb"))
    #tf_idf = get_tfidf_list(docs_for_tfidf)
    #tfidf_dict = dict(zip(tf_idf.index, tf_idf.values))
    cpt_per_utt, cpt_linear = get_concepts(data, cn, stopwords, vocab)
    cpt_ctx, cpt_res = split_cpt(cpt_per_utt, 1)
    print("Processing completed.")
    full_matrix = cn.get_all(vocab, vocab_full, stopwords)
    """
    filtered_cpt_linear = get_topK_by_tfidf(cpt_per_utt, cpt_linear, tfidf_dict, 20)
    expanded_cpt_linear = []
    indexes = []
    for i in range(len(filtered_cpt_linear)):
        expanded, index = cn.expand_list_k(filtered_cpt_linear[i], tfidf_dict, 10)
        expanded_cpt_linear.append(expanded)
        indexes.append(index)
    #result = [[cn.expand_form_of(utt) for utt in line] for line in cpt_per_utt]
    #print(sum([len(line) for line in result]) / len(result))
    expanded = expand_by_path(cpt_ctx, 1000, cn, vocab, stopwords)
    f_1 = [expanded[i] + cpt_ctx[i] for i in range(len(expanded))]
    print("Expand path-1 completed.")
    expanded = expand_by_path(f_1, 1000, cn, vocab, stopwords)
    f_2 = [expanded[i] + f_1[i] for i in range(len(expanded))]
    print("Expand path-2 completed.")
    print(guide_rate_concept(cpt_res, f_2))
    f_2 = [[word for word in f_2[i] if word not in cpt_ctx[i]] for i in range(len(f_2))]
    write_file('valid_3.tsv', data, cpt_per_utt, f_2, cpt_res=cpt_res)
    """

#main()
