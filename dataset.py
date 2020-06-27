import config
import torch
from torch.utils.data import Dataset


class T5Dataset(Dataset):
    def __init__(self, text, selected_text, sentiment):
        self.text = text
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        selected_text = self.selected_text[item]
        sentiment = self.sentiment[item]

        input_text = 'question: ' + sentiment + ' context: ' + text
        input_selected_text = selected_text + ' </s>'

        input_text_encoded = self.tokenizer.encode_plus(input_text, max_length=config.text_max_length,
                                                        pad_to_max_length=True)
        input_selected_text_encoded = self.tokenizer.encode_plus(input_selected_text,
                                                                 max_length=config.selected_text_max_length,
                                                                 pad_to_max_length=True)

        return {
            'input_ids': torch.tensor(input_text_encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(input_text_encoded['attention_mask'], dtype=torch.long),
            'selected_text_ids': torch.tensor([0] + input_selected_text_encoded['input_ids'], dtype=torch.long),
            'selected_text': selected_text
        }
