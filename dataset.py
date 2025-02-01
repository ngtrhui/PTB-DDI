from utils import read_file
from transformers import AutoTokenizer
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch


#-------------------------------------Process Drugbank dataset------------------------------
def process_drugbank_dataset():
    path_drugbank = "./datasets/DrugBank/ddi_total.csv"
    path_train = "./datasets/DrugBank/drugbank_train/raw/train_drugbank_smiles.csv"
    path_test = "./datasets/DrugBank/drugbank_test/raw/test_drugbank_smiles.csv"
    path_train_new = "./datasets/DrugBank/drugbank_train/raw/train_drugbank_smiles_new.csv"
    path_test_new = "./datasets/DrugBank/drugbank_test/raw/test_drugbank_smiles_new.csv"

    drugbank_df = pd.read_csv(path_drugbank, header=None, skiprows=1)
    train_df, test_df = train_test_split(drugbank_df, test_size=0.2, random_state=1234, shuffle=True)
    dnames = ['Drug1_ID', 'Drug1_SMILES', 'Drug2_ID', 'Drug2_SMILES', 'Label_Multi', 'label']
    train_df.to_csv(path_train, index=False, header=dnames)
    test_df.to_csv(path_test, index=False, header=dnames)

    # Extract desired columns
    train_extracted = train_df.iloc[:, [5, 1, 3]]
    test_extracted = test_df.iloc[:, [5, 1, 3]]

    # Rename columns
    dnames_new = ["label", "smile1", "smile2"]

    # Save extracted data to new CSV files
    train_extracted.to_csv(path_train_new, index=False, header=dnames_new)
    test_extracted.to_csv(path_test_new, index=False, header=dnames_new)

# process_drugbank_dataset()
#-----------------------------------------------------------------------------------------

def value_count():
    path_train_new = "./datasets/drugbank/drugbank_train/raw/train_drugbank_smiles_new.csv"
    path_test_new = "./datasets/drugbank/drugbank_test/raw/test_drugbank_smiles_new.csv"
    drugbank_train_new = read_file(path_train_new)
    drugbank_test_new = read_file(path_test_new)
    print("drugbank_train:\n")
    print(drugbank_train_new[0].value_counts())
    print("drugbank_test:\n")
    print(drugbank_test_new[0].value_counts())

class DDIDataset(InMemoryDataset):
    def __init__(self, root='/tmp', path='', transform=None, pre_transform=None):
        self.path = path
        self.model_name = "seyonec/ChemBERTa-zinc-base-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # root is required for save preprocessed data, default is '/tmp'
        super(DDIDataset, self).__init__(root, transform, pre_transform)

        # self.processed_paths is somehow defined from processsed_file_names(self) in InMemoryDataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process_1D(root)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process_1.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_idx_split(self, data_size, train_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:])
        split_dict = {'train': train_idx, 'valid': val_idx}
        return split_dict

    def process_1D(self, root):
        # file_path = 'D:/Documents/nckh/code/PTB-DDI/datasets/drugbank/drugbank_train/raw/train_drugbank_smiles_new.csv'
        # df1 = pd.read_csv(file_path)
        df1 = pd.read_csv('datasets/drugbank/drugbank_train/raw/train_drugbank_smiles_new.csv')

        data_list = []
        data_len = len(df1)

        for i in range(data_len):
            print('Tokenize SMILES to: {}/{}'.format(i + 1, data_len))

            smile1 = df1.loc[i, 'smile1']
            smile2 = df1.loc[i, 'smile2']

            # --------------------------- 1D smiles: smile 1 & 2 ------------------------------------
            train_encoding1 = self.tokenizer(smile1, padding="max_length", truncation=True)
            train_encoding2 = self.tokenizer(smile2, padding="max_length", truncation=True)

            label = df1.loc[i, 'label']
            label = float(label)
            label = torch.tensor(label)

            input_ids1 = train_encoding1['input_ids']
            input_ids1 = torch.tensor(input_ids1)
            attention_mask1 = train_encoding1['attention_mask']
            attention_mask1 = torch.tensor(attention_mask1)

            input_ids2 = train_encoding2['input_ids']
            input_ids2 = torch.tensor(input_ids2)
            attention_mask2 = train_encoding2['attention_mask']
            attention_mask2 = torch.tensor(attention_mask2)

            data = DATA(ids1=input_ids1, mask1=attention_mask1,
                        y=label,
                        ids2=input_ids2, mask2=attention_mask2,
                        )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        # save preprocessed data:
        self._process()
        torch.save((data, slices), self.processed_paths[0])
