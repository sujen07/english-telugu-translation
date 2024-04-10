import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer


def load_data(path):
    english_sentences = []
    telugu_sentences = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            en_text, te_text = line.strip().split('++++$++++')
            english_sentences.append(en_text)
            telugu_sentences.append(te_text)
    return english_sentences, telugu_sentences

def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])



def tokenize(text, vocab, tokenizer):
    return [vocab['<bos>']] + [vocab[token] for token in tokenizer(text)] + [vocab['<eos>']]

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]
        src_numericalized = tokenize(src_sentence, self.src_vocab, self.src_tokenizer)
        trg_numericalized = tokenize(trg_sentence, self.trg_vocab, self.trg_tokenizer)
        return torch.tensor(src_numericalized), torch.tensor(trg_numericalized)


def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item.clone().detach())
        trg_batch.append(trg_item.clone().detach())
    src_batch = pad_sequence(src_batch, padding_value=eng_vocab['<pad>']).transpose(0,1)
    trg_batch = pad_sequence(trg_batch, padding_value=te_vocab['<pad>']).transpose(0,1)
    return src_batch, trg_batch



en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
te_tokenizer = get_tokenizer('basic_english')
eng_sentences, telugu_sentences = load_data('english_telugu_data.txt')
eng_vocab = build_vocab(eng_sentences, en_tokenizer)
te_vocab = build_vocab(telugu_sentences, te_tokenizer)

if __name__ == '__main__':
    dataset = TranslationDataset(eng_sentences, telugu_sentences, eng_vocab, te_vocab, en_tokenizer, te_tokenizer)

    BATCH_SIZE = 64
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)