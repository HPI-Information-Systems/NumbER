from scipy.spatial import distance
import random
import numpy as np
import torch
import torch.nn.functional as F

#https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
def contrastive_loss(pred, labels, t=0.07):
    # positive_tuples, negative_tuples = calculatue_tuples(pred, labels)
    # print("positive_tuples", positive_tuples)
    # print("negative_tuples", negative_tuples)
    # positive_tuples = torch.stack(positive_tuples).to('cuda')
    # negative_tuples = torch.stack(negative_tuples).to('cuda')
    # #.clone().cpu().detach().numpy()
    # positive_distances = [distance.cosine(tuple[0], tuple[1]) for tuple in positive_tuples]
    # negative_distances = [distance.cosine(tuple[0], tuple[1]) for tuple in negative_tuples]
    positive_firsts, positive_seconds, negative_firsts, negative_seconds = calculatue_tuples(pred, labels)
    positive_firsts = torch.stack(positive_firsts).to('cuda')
    positive_seconds = torch.stack(positive_seconds).to('cuda')
    negative_firsts = torch.stack(negative_firsts).to('cuda')
    negative_seconds = torch.stack(negative_seconds).to('cuda')

    # positive_similarities = F.cosine_similarity(positive_firsts[i], positive_seconds[i], dim=0) for i in range(len(positive_firsts))]
    # negative_similarities = F.cosine_similarity(negative_firsts[i], negative_seconds[i], dim=0) for i in range(len(negative_firsts))]
    #print(positive_firsts)
    positive_similarities = F.cosine_similarity(positive_firsts, positive_seconds, dim=1)
    negative_similarities = F.cosine_similarity(negative_firsts, negative_seconds, dim=1)
    #positive_similarities = [1 - distance for distance in positive_distances]
    #negative_similarities = [1 - distance for distance in negative_distances]
    t = 0.07
    #print("positive_similarities", positive_similarities)
    #print("negative_similarities", negative_similarities)
    v = torch.cat((positive_similarities, negative_similarities))
    #average of positive similarities
    print("positive", torch.mean(positive_similarities))
    print("negative", torch.mean(negative_similarities))
    exp = torch.exp(v)
    #softmax_out = exp/np.sum(exp)
    logits = torch.cat((positive_similarities, negative_similarities))/t
    exp = torch.exp(logits)
    loss = - torch.log(exp[0]/torch.sum(exp))
    return loss
    #return torch.from_numpy(np.array(loss))#.to('cuda')
    #return loss

def calculatue_tuples(pred, labels):
    positive_firsts = []
    positive_seconds = []
    negative_firsts = []
    negative_seconds = []
    for idx, item in enumerate(pred):
        label = labels[idx]
        match_indices = [i for i, x in enumerate(labels) if x == label]
        non_match_indices = [i for i, x in enumerate(labels) if x != label]
        random_match_index = random.choice(match_indices)
        random_non_match_index = random.choice(non_match_indices)
        positive_firsts.append(item)
        positive_seconds.append(pred[random_match_index])
        negative_firsts.append(item)
        negative_seconds.append(pred[random_non_match_index])
    return (positive_firsts, positive_seconds, negative_firsts, negative_seconds)


def calculatue_tuples_old(pred, labels):
    positive_tuples = []
    negative_tuples = []
    for idx, item in enumerate(pred):
        label = labels[idx]
        match_indices = [i for i, x in enumerate(labels) if x == label]
        non_match_indices = [i for i, x in enumerate(labels) if x != label]
        random_match_index = random.choice(match_indices)
        random_non_match_index = random.choice(non_match_indices)
        #tuples.append([(item, pred[random_match_index]), (item, pred[random_non_match_index])])
        positive_tuples.append((item, pred[random_match_index]))
        negative_tuples.append((item, pred[random_non_match_index]))
        #tuples.append((item, pred[random_match_index]))
        #tuples.append((item, pred[random_non_match_index]))
    # digit_indices = [np.where(labels == i)[0] for i in range(10)]
    # labels_output = []
    # for idx1 in range(len(pred)):
    #     x1 = pred[idx1]
    #     label1 = labels[idx1]
    #     idx2 = random.choice(digit_indices[label1])
    #     x2 = pred[idx2]
    #     tuples.append((x1, x2))
    #     labels_output.append(1)
    #     idx2 = random.choice(digit_indices[(label1 + 1)%10])
    #     x2 = pred[idx2]
    #     tuples.append((x1, x2))
    #     labels_output.append(0)
    return positive_tuples, negative_tuples
        
        