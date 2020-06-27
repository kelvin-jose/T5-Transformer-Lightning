from transformers import T5Tokenizer

text_max_length = 1024
selected_text_max_length = 56
TOKENIZER = T5Tokenizer.from_pretrained('t5-base')
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.001
BATCH_TRAIN = 64
BATCH_TEST = 32
EPOCHS = 10