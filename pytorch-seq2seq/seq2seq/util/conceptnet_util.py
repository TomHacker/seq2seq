import json
import numpy as np
from collections import defaultdict
import pickle
import os


class ConceptNet:
    def __init__(self, path, vocab=None):
        self.cpt_dict = json.load(open(path))
        if vocab:
            adjacency_matrix = defaultdict(list)
            for word in vocab:
                adjacency_matrix[word] = self.find_adjacent(word, vocab)
            print("Initialization finished.")
            self.adjacency_matrix = adjacency_matrix
        else:
            self.adjacency_matrix = None

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

    def get_adjacency_matrix(self, vocab, stopwords=None, full_vocab=None):
        concepts = list(vocab.keys())
        final_res = np.zeros((len(vocab) + 2, len(vocab) + 2))
        print(final_res.shape)
        for i, cpt in enumerate(concepts):
            if self.adjacency_matrix:
                adjacent = self.adjacency_matrix[cpt]
            else:
                adjacent = self.find_adjacent(cpt, vocab)
            index = vocab[cpt]
            num_edges = 0
            for tmp in adjacent:
                if tmp not in stopwords:
                    num_edges += 1
            if num_edges == 0:
                continue
            weight = 1 / num_edges
            for tmp in adjacent:
                if tmp in stopwords:
                    continue
                final_res[index][vocab[tmp]] += weight
        return final_res

    def get_all(self, vocab, vocab_full, stopwords=None):
        if not stopwords:
            stopwords = [word.strip() for word in open('../../../ConceptNet/stopword.txt').readlines()]
            stopwords = {word: 1 for word in stopwords}
        A = self.get_adjacency_matrix(vocab, stopwords)
        print("Adacency matrix processed.")
        print(os.path.exists("./seq2seq/util/A_square_1"))
        if not os.path.exists("A_square_1"):
            A_2 = np.dot(A, A)
            print("A square processed.")
            pickle.dump(A_2[:10000, :], open("A_square_1", "wb"))
            pickle.dump(A_2[10000:20000, :], open("A_square_2", "wb"))
            pickle.dump(A_2[20000:, :], open("A_square_3", "wb"))
            print("Save file successfully!")
        else:
            part_1 = pickle.load(open("A_square_1", "rb"))
            part_2 = pickle.load(open("A_square_2", "rb"))
            part_3 = pickle.load(open("A_square_3", "rb"))
            A_2 = np.concatenate((part_1, part_2, part_3), axis=0)
            print("Successfully loaded.")
        full = np.zeros((len(vocab_full), len(vocab_full)))
        full[:len(A), :len(A)] = (A + A_2)
        print(full.shape)
        return full

    def expand_list_by_path(self, concept, k, vocab=None, stopword=None, return_dict=False):
        time_dict = {}
        concept_dict = {word: 1 for word in concept}
        for cpt in concept:
            if self.adjacency_matrix:
                adjacent = self.adjacency_matrix[cpt]
            else:
                adjacent = self.find_adjacent(cpt, vocab)
            num_edges = len(adjacent)
            if num_edges == 0:
                continue
            weight = 1 / num_edges
            for tmp in adjacent:
                if tmp in stopword:
                    continue
                if tmp in concept_dict and not return_dict:
                    continue
                if tmp not in time_dict:
                    time_dict[tmp] = weight
                else:
                    time_dict[tmp] += weight

        res_1 = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)
        res_1 = [word[0] for word in res_1]
        res_dict = {word: 1 for word in res_1}
        #jump_1 = res_1[:k]
        new_time_dict = {}
        for cpt in res_1:
            if self.adjacency_matrix:
                adjacent = self.adjacency_matrix[cpt]
            else:
                adjacent = self.find_adjacent(cpt, vocab)
            num_edges = len(adjacent)
            if num_edges == 0:
                continue
            weight = 1 / num_edges
            for tmp in adjacent:
                if tmp in stopword:
                    continue
                if tmp not in new_time_dict:
                    new_time_dict[tmp] = weight * time_dict[cpt]
                else:
                    new_time_dict[tmp] += weight * time_dict[cpt]
        for cpt in res_1:
            if cpt in new_time_dict:
                new_time_dict[cpt] += time_dict[cpt]
            else:
                new_time_dict[cpt] = time_dict[cpt]

        if return_dict:
            tmp_dict = {}
            for time in time_dict.values():
                if time not in tmp_dict:
                    tmp_dict[time] = 1
                else:
                    tmp_dict[time] += 1
            return tmp_dict

        res_2 = sorted(new_time_dict.items(), key=lambda x: x[1], reverse=True)
        res_2 = [word[0] for word in res_2]
        return list(set(res_2[:k]))
        #return res_1[:k]

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

