import torch
import torch.utils.data as data
import spacy
from src import config

class TranslationDataset(data.Dataset):
    ''' 
    A placeholder dataset.
    In a real scenario, you would load your data from files,
    and this class would handle tokenization and numericalization.
    '''
    def __init__(self, src_lang="en_core_web_sm", trg_lang="de_core_news_sm"):
        self.spacy_src = spacy.load(src_lang)
        self.spacy_trg = spacy.load(trg_lang)

        # Placeholder data
        self.data = [
            ("This is a sentence.", "Das ist ein Satz."),
            ("Attention is all you need.", "Aufmerksamkeit ist alles, was du brauchst."),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src_text, trg_text = self.data[index]

        # In a real implementation, you would build a vocabulary
        # and convert tokens to indices.
        src_tokens = [tok.text for tok in self.spacy_src.tokenizer(src_text)]
        trg_tokens = [tok.text for tok in self.spacy_trg.tokenizer(trg_text)]

        # For this example, we'll just return the text.
        # A full implementation would return tensors of indices.
        return src_tokens, trg_tokens

def get_dataloader():
    dataset = TranslationDataset()
    
    # The collate_fn would handle padding and creating tensors.
    # This is a simplified version.
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(src_sample)
            trg_batch.append(trg_sample)
        return src_batch, trg_batch

    dataloader = data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    return dataloader
