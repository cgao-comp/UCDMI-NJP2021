from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
import torch
from utils.f1score import cluster_acc



def convertMatrix_listlabel(labels):
    """

    :param labels: label matrix
    :return: label list
    """
    node_num = labels.shape[0]
    label_num = labels.shape[1]
    label_list = []
    for i in range(node_num):
        for j in range(label_num):
            if labels[i,j] == 1 or j==label_num-1:
                label_list.append(j)
                break
    return label_list

def assement_directly(labels, pre):
    labels = convertMatrix_listlabel(labels)
    NMI = metrics.normalized_mutual_info_score(labels, pre)
    print("the valve of NMI is：{}".format(NMI))
def assement_result(labels,embeddings,k):
    embeddings = torch.squeeze(embeddings, 0)
    embeddings = embeddings.numpy()
    labels = convertMatrix_listlabel(labels)
    origin_cluster = labels
    a = 0
    sum = 0
    sumF1score = 0
    sumARI = 0
    sumAccuracy = 0
    reapeats = 3
    while a < reapeats:
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings)

        c = y_pred.T
        epriment_cluster = c ;
        NMI = metrics.normalized_mutual_info_score(origin_cluster, epriment_cluster)
        accuracy, F1_score = cluster_acc(origin_cluster, epriment_cluster)
        ARI = adjusted_rand_score(origin_cluster, epriment_cluster)
        sum = sum + NMI
        sumF1score = sumF1score + F1_score
        sumARI = sumARI + ARI
        sumAccuracy = accuracy + sumAccuracy
        a = a + 1
    average_NMI = sum / reapeats
    average_F1score = sumF1score / reapeats
    average_ARI = sumARI / reapeats
    average_Accuracy = sumAccuracy / reapeats
    return average_NMI, average_F1score, average_ARI, average_Accuracy