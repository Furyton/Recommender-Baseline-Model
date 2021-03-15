import numpy
from tqdm import tqdm


# popularity is a dict
# observed, answer, candidates are list

def test(popularity: dict, observed_list: list, answer_list: list, candidates_list: list):
    predict_rank = []

    for observed, answer, candidates in tqdm(zip(observed_list, answer_list, candidates_list)):
        score_list = [key for key, value in sorted({key: popularity[key] for key in answer + candidates}.items(),
                                                   key=lambda item: -item[1])]

        predict_rank.append(score_list.index(answer[0]))

    return predict_rank
