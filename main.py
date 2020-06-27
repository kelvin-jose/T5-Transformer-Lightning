import config
from model import T5
import pytorch_lightning as light

train_params = dict(
        max_epochs=config.EPOCHS,
        early_stop_callback=False,
        gradient_clip_val=1.0,
    )

trainer = light.trainer.trainer.Trainer(**train_params)
model = T5()
trainer.fit(model)


