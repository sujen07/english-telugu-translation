
from models import EncoderRNN, DecoderRNN
from prepare_data import *
from torch.utils.data import DataLoader
import torch




dataset = TranslationDataset(eng_sentences, telugu_sentences, eng_vocab, te_vocab, en_tokenizer, te_tokenizer)

BATCH_SIZE = 64
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# Initialize models and optimizers
eng_size = len(eng_vocab)
te_size = len(te_vocab)

learning_rate = 0.0001
hidden_size = 65

encoder = EncoderRNN(eng_size, hidden_size)
decoder = DecoderRNN(hidden_size, te_size, te_vocab)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


criterion = torch.nn.NLLLoss()

print(len(dataloader))
for src,trg in dataloader:

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    max_length = trg.shape[-1]
    output, hidden_state = encoder(src)
    decoder_out, decoder_hidden = decoder(output, hidden_state, max_length=max_length)

    decode_out = decoder_out.view(-1, decoder_out.shape[-1]) # Reshape to [batch_size * seq_len, vocab_size]
    trg = trg.reshape(-1)  # Flatten to [batch_size * seq_len]
    loss = criterion(decode_out, trg)
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()
    print(f'loss: {loss.item()}')


sentence = 'What is your name?'
inp = tokenize(sentence, eng_vocab, en_tokenizer)
tensor_inp = torch.tensor(inp).unsqueeze(0)

with torch.no_grad():
    enc_out,enc_hidden = encoder(tensor_inp)
    decoder_out, decoder_hidden = decoder(enc_out, enc_hidden)

max_indices = torch.argmax(decoder_out, dim=-1)

for i in max_indices[0]:
    te_vocab.lookup_token(i.item())


