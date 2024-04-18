from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from prepare_data import *
from torch.utils.data import DataLoader
import torch
import tqdm




dataset = TranslationDataset(eng_sentences, telugu_sentences, eng_vocab, te_vocab)

BATCH_SIZE = 128
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Initialize models and optimizers
eng_size = len(eng_vocab)
te_size = len(te_vocab)

learning_rate = 0.001
hidden_size = 128

encoder = EncoderRNN(eng_size, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, te_size, te_vocab).to(device)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


criterion = torch.nn.NLLLoss(ignore_index=eng_vocab['<pad>']).to(device)

def eval_sent(sentence):
    inp = tokenize(sentence, eng_vocab)
    tensor_inp = torch.tensor(inp).unsqueeze(0)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        enc_out,enc_hidden = encoder(tensor_inp.to(device))
        decoder_out, decoder_hidden,_ = decoder(enc_out, enc_hidden)

    max_indices = torch.argmax(decoder_out, dim=-1)

    l = []
    for i in max_indices[0]:
        l.append(te_vocab.lookup_token(i.item()))

    print(''.join(l))
    return l

print(len(dataloader))
for src,trg in tqdm.tqdm(dataloader):

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    max_length = trg.shape[-1]
    output, hidden_state = encoder(src.to(device))
    decoder_out, decoder_hidden,_ = decoder(output, hidden_state, trg, max_length=max_length)

    decode_out = decoder_out.reshape(-1, decoder_out.shape[-1]) # Reshape to [batch_size * seq_len, vocab_size]
    trg = trg.reshape(-1)  # Flatten to [batch_size * seq_len]
    loss = criterion(decode_out, trg.to(device))
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()
    print(f'loss: {loss.item()}')




