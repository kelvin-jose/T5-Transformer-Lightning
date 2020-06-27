import config
import numpy as np
from transformers import AdamW
from utils import get_train_dataloaders
from utils import get_val_dataloaders
from utils import get_num_train_steps
from utils import jaccard
import pytorch_lightning as light
from transformers import T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup


class T5(light.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.model.init_weights()

    def forward(self, input_ids, attention_mask, lm_labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, lm_labels=lm_labels)

    def train_dataloader(self):
        train_dl = get_train_dataloaders()
        return train_dl

    def val_dataloader(self):
        val_dl = get_val_dataloaders()
        return val_dl

    def training_step(self, batch, batch_nb):
        y = batch['selected_text_ids']
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == 0] = -100
        output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], lm_labels=lm_labels)
        tensorboard_logs = {'train_loss': output[0]}
        return {'loss': output[0], 'log': tensorboard_logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)
        self.opt = optimizer
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=get_num_train_steps()
        )
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.scheduler.step()

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=config.selected_text_max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        preds = [
            config.TOKENIZER.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]

        return {"preds": preds}

    def validation_step(self, batch, batch_idx):
        target_text = batch['selected_text']

        preds = self.test_step(batch, batch_idx)
        preds_text = preds["preds"]
        jaccard_score = [jaccard(p, t) for p, t in zip(preds_text, target_text)]

        return {"jaccard_score": jaccard_score}


    def validation_end(self, outputs):
        jaccard_scores = sum([x["jaccard_score"] for x in outputs], [])
        avg_jaccard_score = np.mean(jaccard_scores)
        tensorboard_logs = { "jaccard_score": avg_jaccard_score}
        return {"avg_jaccard_score": avg_jaccard_score, "log": tensorboard_logs}


