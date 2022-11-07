from os import truncate
from unittest.util import _MAX_LENGTH
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
def test(corpus):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    preprocess_text = corpus.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt",max_length=512,padding='max_length',truncation=True).to(device)
    summary_ids = model.generate(tokenized_text,num_beams=2,no_repeat_ngram_size=3,length_penalty=2.0,min_length=80,max_length=1000,early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return output


if __name__ == '__main__':
    test()