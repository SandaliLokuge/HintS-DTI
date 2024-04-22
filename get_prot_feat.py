#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df1 = pd.read_csv("/N/slate/sdlokuge/nlp/project/DLM-DTI_hint-based-learning/data/train_dataset.csv")
df2 = pd.read_csv("/N/slate/sdlokuge/nlp/project/DLM-DTI_hint-based-learning/data/valid_dataset.csv")
df3 = pd.read_csv("/N/slate/sdlokuge/nlp/project/DLM-DTI_hint-based-learning/data/test_dataset.csv")

df = pd.concat([df1, df2, df3]).reset_index(drop=True)
df = df.loc[:, "Target Sequence"].drop_duplicates().reset_index(drop=True)
df


# In[ ]:


import pickle
import pandas as pd
import transformers
from tqdm import tqdm
from transformers import AutoModel, BertTokenizer, RobertaTokenizer

prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
prot_encoder = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to("cuda")
prot_encoder.eval()


# In[ ]:


import os
os.makedirs('prot_feat', exist_ok=True)

for max_length in [545]:
    results = {}

    for data in tqdm(df, total=len(df)):
        seq = prot_tokenizer(" ".join(data), max_length=max_length + 2, truncation=True, return_tensors="pt").to("cuda")
        a = prot_encoder(**seq)
        a = a.last_hidden_state.detach().to("cpu")
        results[data[:20]] = a[:, 0]

    with open(f"prot_feat/{max_length}_cls.pkl", "wb") as f:
        pickle.dump(results, f)


# In[ ]:




