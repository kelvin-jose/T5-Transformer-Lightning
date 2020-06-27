import config
import pandas as pd
from dataset import T5Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/train.csv')
df.dropna(inplace=True)

df_train, df_val = train_test_split(df, test_size=0.12)
df_val.reset_index(inplace=True, drop=True)
df_train.reset_index(inplace=True, drop=True)

def get_train_dataloaders():
    train_dataset = T5Dataset(df_train['text'], df_train['selected_text'], df_train['sentiment'])
    train_dataloader = DataLoader(train_dataset, config.BATCH_TRAIN)
    return train_dataloader

def get_val_dataloaders():
    val_dataset = T5Dataset(df_val['text'], df_val['selected_text'], df_val['sentiment'])
    val_dataloader = DataLoader(val_dataset, config.BATCH_TEST)
    return val_dataloader
def get_num_train_steps():
    return int(len(df_train) / config.BATCH_TRAIN * config.EPOCHS)

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
