import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoModel, BertTokenizer, RobertaTokenizer
from transformers import BertConfig, BertModel


def define_mol_encoder(is_freeze=True):
    mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mol_encoder = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    if is_freeze:
        for param in mol_encoder.embeddings.parameters():
            param.requires_grad = False

        for layer in mol_encoder.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

    return mol_tokenizer, mol_encoder


def define_prot_encoder(
    max_length,
    hidden_size=512,
    num_hidden_layer=4,
    num_attention_heads=4,
    intermediate_size=2048,
    hidden_act="gelu",
    pad_token_id=0,
):
    prot_tokenizer = BertTokenizer.from_pretrained(
        "Rostlab/prot_bert_bfd", do_lower_case=False
    )
    # prot_encoder = AutoModel.from_pretrained("Rostlab/prot_bert")

    config = BertConfig(
        vocab_size=prot_tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layer,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length + 2,
        type_vocab_size=1,
        pad_token_id=0,
        position_embedding_type="absolute",
    )

    prot_encoder = BertModel(config)

    return prot_tokenizer, prot_encoder


class DTI_SiameseNetwork(nn.Module):
    def __init__(
        self,
        mol_encoder,
        prot_encoder,
        is_learnable_lambda=True,
        fixed_lambda=-1,
        hidden_dim=512,
        mol_dim=768,
        prot_dim=512,
        device_no=0,
    ):
        super().__init__()
        self.mol_encoder = mol_encoder
        self.prot_encoder = prot_encoder
        self.is_learnable_lambda = is_learnable_lambda

        if self.is_learnable_lambda and fixed_lambda == -1:
            self.lambda_ = torch.nn.Parameter(
                torch.rand(1).to(f"cuda:{device_no}"), requires_grad=True
            )
        elif self.is_learnable_lambda == False and (
            (fixed_lambda >= 0) and (fixed_lambda <= 1)
        ):
            lambda_ = torch.ones(1) * fixed_lambda
            self.lambda_ = lambda_.to(f"cuda:{device_no}")
        print(f"Initial lambda parameter: {self.lambda_}")

        self.molecule_align = nn.Sequential(
            nn.LayerNorm(mol_dim), nn.Linear(mol_dim, hidden_dim, bias=False)
        )

        self.protein_align_teacher = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, hidden_dim, bias=False)
        )

        self.protein_align_student = nn.Sequential(
            nn.LayerNorm(prot_dim), nn.Linear(prot_dim, hidden_dim, bias=False)
        )

        # original code in DLM-DTI
        # self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        # self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        # self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.cls_out = nn.Linear(hidden_dim, 1)

        # changed - for siamese network
        layer1_out_dim = int(hidden_dim/2)
        layer2_out_dim = int(hidden_dim/4)
        self.fc1 = nn.Linear(hidden_dim, layer1_out_dim)
        self.fc2 = nn.Linear(layer1_out_dim, layer2_out_dim)
        self.dist_fc = nn.Linear(layer2_out_dim,out_features=1)

    # added - for siamese network
    def forward_once(self, x, drop_out):
        # Forward pass 
        x = F.dropout(F.relu(self.fc1(x)), drop_out)
        x = F.dropout(F.relu(self.fc2(x)), drop_out)
        return x

    def forward(self, SMILES, FASTA, prot_feat_teacher):
        mol_feat = self.mol_encoder(**SMILES).last_hidden_state[:, 0]
        prot_feat = self.prot_encoder(**FASTA).last_hidden_state[:, 0]

        mol_feat = self.molecule_align(mol_feat)
        prot_feat = self.protein_align_student(prot_feat)
        prot_feat_teacher = self.protein_align_teacher(prot_feat_teacher).squeeze(1)

        if self.is_learnable_lambda == True:
            lambda_ = torch.sigmoid(self.lambda_)
        elif self.is_learnable_lambda == False:
            lambda_ = self.lambda_.detach()

        merged_prot_feat = lambda_ * prot_feat + (1 - lambda_) * prot_feat_teacher

        ## original code in DLM-DTI
        # x = torch.cat([mol_feat, merged_prot_feat], dim=1)
        # x = F.dropout(F.gelu(self.fc1(x)), 0.1)
        # x = F.dropout(F.gelu(self.fc2(x)), 0.1)
        # x = F.dropout(F.gelu(self.fc3(x)), 0.1)
        # cls_out = self.cls_out(x).squeeze(-1)
        # return cls_out, lambda_.mean()
    
        ## changed - for siamese network
        prot_output = self.forward_once(merged_prot_feat, 0.3)
        drug_output = self.forward_once(mol_feat, 0.3)
        similarity = torch.abs(prot_output - drug_output) ** 2
        cls_dist = self.dist_fc(similarity).squeeze(1)

        return cls_dist, lambda_.mean()
