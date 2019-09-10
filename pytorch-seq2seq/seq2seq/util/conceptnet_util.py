import json
import numpy as np


class ConceptNet:
    def __init__(self, path):
        self.cpt_dict = json.load(open(path))

    def find_adjacent(self, concept, vocab):
        if concept not in self.cpt_dict.keys():
            return []
        raw = list(set(list(self.cpt_dict[concept].keys())))
        return [word for word in raw if word in vocab]

    def find_adjacent_k(self, concept, tf_idf, k):
        adjacent = list(set(self.cpt_dict[concept]))
        adjacent = [word for word in adjacent if word in tf_idf]
        result = sorted(adjacent, key=lambda x: tf_idf[x], reverse=True)
        if len(result) <= k:
            return result
        else:
            return result[:k]

    def find_adjacent_by_path(self, concept, k):
        adjacent = list(set(self.cpt_dict[concept]))

    def find_form_of(self, concept):
        res = []
        for cpt in self.cpt_dict[concept].keys():
            for relation in self.cpt_dict[concept][cpt]:
                if relation[0] == "INV-FormOf" or relation[0] == "FormOf":
                    res.append(cpt)
        if concept not in res:
            res.append(concept)
        return res

    def expand_list(self, concept_list, vocab=False):
        res = []
        index = []
        for cpt in concept_list:
            res.extend(self.find_adjacent(cpt, vocab))
            #index.append(len(res))
        return list(set(res))

    def expand_list_k(self, concept_list, tf_idf, k):
        res = []
        indexes = []
        for cpt in concept_list:
            expand = self.find_adjacent_k(cpt, tf_idf, k)
            res.extend(expand)
            indexes.append(len(res))
        return res, indexes

    def expand_corpus(self, corpus, vocab=False):
        return [self.expand_list(line, vocab) for line in corpus]

    def expand_list_by_path(self, concept, k, vocab=None, stopword=None):
        time_dict = {}
        for cpt in concept:
            adjacent = self.find_adjacent(cpt, vocab)
            for tmp in adjacent:
                if tmp in stopword:
                    continue
                if tmp in concept:
                    continue
                if tmp not in time_dict:
                    time_dict[tmp] = 1
                else:
                    time_dict[tmp] += 1
        res = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)
        res = [word[0] for word in res if '_' not in word[0]]
        return res[:k]

    def expand_form_of(self, concept_list):
        res = []
        for cpt in concept_list:
            res.extend(self.find_form_of(cpt))
        return list(set(res))

    def get_max_score(self, cpt1, cpt2):
        return max([rel[1] for rel in self.cpt_dict[cpt1][cpt2]])

    def get_topK(self, cpt, K):
        res = []
        for candidate in self.cpt_dict[cpt].keys():
            res.append((candidate, self.get_max_score(cpt, candidate)))
        return sorted(res, key=lambda x: x[1], reverse=True)[:K]

    def get_randomK(self, cpt, K):
        res = []
        for candidate in self.cpt_dict[cpt].keys():
            res.append((candidate, self.get_max_score(cpt, candidate)))
        if K > len(res):
            return res
        else:
            final_res = []
            index = np.random.randint(len(res), size=K)
            for id in index:
                final_res.append(res[id])
            return final_res

