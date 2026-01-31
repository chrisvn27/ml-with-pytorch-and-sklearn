from transformers import pipeline, set_seed

generator = pipeline('text-generation', model= 'gpt2')
set_seed(123)

print(generator("Thomas Mann is", max_new_tokens=20,
          num_return_sequences=3))

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Let us encode this sentence"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)

from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')
output = model(**encoded_input)
print(output['last_hidden_state'].shape)

