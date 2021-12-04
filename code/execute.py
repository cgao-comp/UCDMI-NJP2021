import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import UCDMI, LogReg
from utils import process, clustering

# dataset = 'cora'
dataset = 'citeseer'
# dataset = 'pubmed'


# training params
batch_size = 1
nb_epochs = 200
patience = 200
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 128
jaccard = 0.3

save_Finally_values_list = []
print("the value of jaccard is:{}".format(jaccard))
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

adj, features, labels, kmeans_labels, idx_train, idx_val, idx_test, graph = process.load_data_our(dataset, jaccard)   ##labels: arraya类型[2708,7]
labels_ori = labels
adj_ori = process.sparse_mx_to_torch_sparse_tensor(adj)

cc_label = process.convert_label2(kmeans_labels)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = UCDMI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
best_nmi = 0
best_epoch = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)


    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits= model(cc_label, features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

    loss = b_xent(logits, lbl)
    embeds_1, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    average_NMI, average_F1score, average_ARI, average_Acc = clustering.assement_result(labels_ori, embeds_1, nb_classes)

    # print('Loss:', loss)
    print("{0}th epoch | loss:{1} | nmi:{2} | acc:{3} | f-score:{4} | ari:{5}".format(epoch, loss, average_NMI, average_Acc, average_F1score, average_ARI))
    if average_NMI > best_nmi:
        best_nmi = average_NMI
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_ucdmi_nmi.pkl')

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_ucdmi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('the {}th epoch is the best epoch'.format(best_epoch))
