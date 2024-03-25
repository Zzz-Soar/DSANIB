import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN

def loss_function(res_pre, labels, vec_mean, vec_cov, beta=1e-3):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(res_pre), 1)
    loss = loss_fct(n, labels)
    KL = 0.5 * torch.sum(vec_mean.pow(2) + vec_cov.pow(2) - 2 * vec_cov.log() - 1)

    return n, loss + beta * KL / res_pre.shape[0]

class DSANIB(nn.Module):
    def __init__(self, **config):
        super(DSANIB, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_emb_dim = config["DRUG"]["NODE_EMBEDDING"]
        drug_filter_dimens = config["DRUG"]["GCN_Dimension"]
        protein_emb_dim = config["PROTEIN"]["EMB_SIZE"]
        protein_filter_dimens = config["PROTEIN"]["CNN_Dimension"]
        filter_size = config["PROTEIN"]["FILTER_SIZE"]
        in_dimensions = config["MLPLayer"]["IN_DIM"]
        hidden_dimensions = config["MLPLayer"]["HIDDEN_DIM"]
        out_dimemnsions = config["MLPLayer"]["OUT_DIMs"]
        outprob = config["MLPLayer"]["OUTPROB"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]

        self.drug_extractor = Drug_GCN_Emb(drug_in_feats, drug_emb_dim, drug_padding, drug_filter_dimens)

        self.protein_extractor = Protein_CNN_Emb(protein_emb_dim, protein_filter_dimens, filter_size, protein_padding)
        self.dual_view_layer = nn.MultiheadAttention(128, 4)

        self.drugPool = nn.MaxPool1d(290)
        self.proteinPool = nn.MaxPool1d(1185)
        self.ib = InformationBottleneck()
        self.decoder = MLPDecoder(in_dimensions, hidden_dimensions,out_dimemnsions, outprob)

    def forward(self, drug_Initial, protein_Initial, mode="train"):
        drugGCN = self.drug_extractor(drug_Initial)
        proteinCNN = self.protein_extractor(protein_Initial)

        drug_linear = drugGCN.permute(1, 0, 2)
        protein_linear = proteinCNN.permute(1, 0, 2)

        drug_att, _ = self.dual_view_layer(drug_linear, protein_linear, protein_linear)
        protein_att, _ = self.dual_view_layer(protein_linear, drug_linear, drug_linear)

        drug_att1, _ = self.dual_view_layer(drug_att, drug_linear, drug_att)
        protein_att1, _ = self.dual_view_layer(protein_att, protein_linear, protein_att)

        drug_att1 = drug_att1.permute(1, 0, 2)
        protein_att1 = protein_att1.permute(1, 0, 2)

        drug1 = drugGCN * 0.5 + drug_att1 * 0.5
        protein1 = proteinCNN * 0.5 + protein_att1 * 0.5

        # 第二层
        drug_linear2 = drug1.permute(1, 0, 2)
        protein_linear2 = protein1.permute(1, 0, 2)

        drug_att2, _ = self.dual_view_layer(drug_linear2, protein_linear2, protein_linear2)
        protein_att2, _ = self.dual_view_layer(protein_linear2, drug_linear2, drug_linear2)

        drug_att3, _ = self.dual_view_layer(drug_att2, drug_linear2, drug_att2)
        protein_att3, _ = self.dual_view_layer(protein_att2, protein_linear2, protein_att2)

        drug_att3 = drug_att3.permute(1, 0, 2)
        protein_att3 = protein_att3.permute(1, 0, 2)

        drug2 = drug1 * 0.5 + drug_att3 * 0.5
        protein2 = protein1 * 0.5 + protein_att3 * 0.5

        # 第三层
        drug_linear3 = drug2.permute(1, 0, 2)
        protein_linear3 = protein2.permute(1, 0, 2)

        drug_att4, _ = self.dual_view_layer(drug_linear3, protein_linear3, protein_linear3)
        protein_att4, _ = self.dual_view_layer(protein_linear3, drug_linear3, drug_linear3)

        drug_att5, _ = self.dual_view_layer(drug_att4, drug_linear3, drug_att4)
        protein_att5, _ = self.dual_view_layer(protein_att4, protein_linear3, protein_att4)

        drug_att5 = drug_att5.permute(1, 0, 2)
        protein_att5 = protein_att5.permute(1, 0, 2)

        drug3 = drug2 * 0.5 + drug_att5 * 0.5
        protein3 = protein2 * 0.5 + protein_att5 * 0.5

        drug = drug3.permute(0, 2, 1)
        protein = protein3.permute(0, 2, 1)

        F_drug = self.drugPool(drug).squeeze(2)
        F_protein = self.proteinPool(protein).squeeze(2)

        pair = torch.cat([F_drug, F_protein], dim=1)
        jointF, vec_mean, vec_cov, = self.ib(pair)

        prob = self.decoder(jointF)
        if mode == "train":
            return prob, vec_mean, vec_cov

class Drug_GCN_Emb(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(Drug_GCN_Emb, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class Protein_CNN_Emb(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(Protein_CNN_Emb, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class InformationBottleneck(nn.Module):
    def __init__(self):
        super(InformationBottleneck, self).__init__()
        # dropout = {0,1~0.5}
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.vec_mean = nn.Linear(1024, 256)
        self.vec_cov = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, pair):
        pair = self.dropout(self.bn1(pair))
        pair = self.bn2(F.relu(self.fc1(pair)))
        pair = self.bn2(F.relu(self.fc2(pair)))

        vec_mean, vec_cov = self.bn3(self.vec_mean(pair)), F.softplus(self.bn3(self.vec_cov(pair)) - 5)
        eps = torch.randn_like(vec_cov)
        jointF = vec_mean + vec_cov * eps
        return jointF, vec_mean, vec_cov

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        #x = self.bn1(F.relu(self.fc1(x)))
        #x = self.bn2(F.relu(self.fc2(x)))
        #x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

