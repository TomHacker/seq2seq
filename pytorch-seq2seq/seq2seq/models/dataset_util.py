import json
from nltk.tokenize import wordpunct_tokenize
from conceptnet_util import ConceptNet
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd


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


def get_concepts(data, cn):
    cpt_per_utt = [[list(set([word for word in sentence if word in cn.cpt_dict and word not in stopwords])) for sentence in dialog] for dialog in data]
    cpt_linear = []
    for dialog in cpt_per_utt:
        cpt = []
        for sent in dialog:
            cpt.extend(sent)
        cpt_linear.append(cpt)
    return cpt_per_utt, cpt_linear


def get_topK_by_tfidf(cpt_per_utt, cpt_linear, tf_idf, K):
    result = []
    for i, dialog in enumerate(cpt_linear):
        in_dict = [word for word in dialog if word in tf_idf]
        res = sorted(in_dict, key=lambda x: tf_idf[x], reverse=True)
        if len(res) > K:
            result.append(res[:K])
        else:
            result.append(res)
        for j, utt in enumerate(cpt_per_utt[i]):
            cpt_per_utt[i][j] = [word for word in cpt_per_utt[i][j] if word in result[-1]]
    return result


def write_file(path, dialogs, concepts):
    with open(path, 'w') as f:
        for i in range(len(dialogs)):
            dialog_str = ""
            concept_str = ""
            for j in range(len(dialogs[i]) -1):
                dialog_str += " ".join(dialogs[i][j]) + " <eou> "
                concept_str += " ".join(concepts[i][j]) + " <eou> "
            response = " ".join(dialogs[i][-1])
            f.write(dialog_str + '\t' + response + '\t' + concept_str + '\n')


path = 'train.json'
cn = ConceptNet("concept_dict.json")
stopwords = [word.strip() for word in open('stopword.txt').readlines()]
data, docs_for_tfidf = load_data(path)
tf_idf = get_tfidf_list(docs_for_tfidf)
tfidf_dict = dict(zip(tf_idf.index, tf_idf.values))
cpt_per_utt, cpt_linear = get_concepts(data, cn)
filtered_cpt_linear = get_topK_by_tfidf(cpt_per_utt, cpt_linear, tfidf_dict, 20)
#result = [[cn.expand_form_of(utt) for utt in line] for line in cpt_per_utt]
#print(sum([len(line) for line in result]) / len(result))
write_file('train_2.tsv', data, cpt_per_utt)
