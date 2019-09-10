import numpy


def judge_same(ngram1, ngram2):
    for i in range(len(ngram1)):
        if ngram1[i] != ngram2[i]:
            return False
    return True


def distinct(candidates, n):
    score_all = 0
    ngrams = []
    cnt = 0
    for line in candidates:
        for i in range(len(line) - n + 1):
            ngrams.append(line[i:i+n])
    for i in range(len(ngrams)):
        for j in range(i+1, len(ngrams)):
            if judge_same(ngrams[i], ngrams[j]):
                cnt += 1
                break
    all_words = sum([len(line) for line in candidates])
    if all_words == 0:
        return 0
    return (len(ngrams) - cnt) / all_words
