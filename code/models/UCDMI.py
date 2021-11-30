import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator



class UCDMI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(UCDMI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.C = nn.Parameter(torch.FloatTensor(2708, 7))
        self.E = nn.Parameter(torch.FloatTensor(7, 128))
        self.I = torch.eye(7)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.C)
        nn.init.xavier_normal_(self.E)

    def forward(self, cc_label, seq1, seq2, adj, sparse, msk, samp_bias1,
                samp_bias2):
        node_num = seq1.size()[1]
        ret_1 = torch.empty(1, node_num)
        ret_2 = torch.empty(1, node_num)
        h_1 = self.gcn(seq1, adj, sparse)

        h_2 = self.gcn(seq2, adj, sparse)
        for i in range(len(cc_label)):
            h_11 = h_1[0, cc_label[i], :]
            h_22 = h_2[0, cc_label[i], :]
            h_11 = torch.unsqueeze(h_11, 0)
            h_22 = torch.unsqueeze(h_22, 0)
            c = self.read(h_11, msk)
            c = self.sigm(c)

            sc_1, sc_2 = self.disc(c, h_11, h_22, samp_bias1, samp_bias2)

            for p in range(len(cc_label[i])):
                ret_1[0, cc_label[i][p]] = sc_1[0, p]
                ret_2[0, cc_label[i][p]] = sc_2[0, p]

        ret = torch.cat((ret_1, ret_2), 1)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
