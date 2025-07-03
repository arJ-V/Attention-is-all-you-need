import torch
import torch.nn as nn
import torch.optim as optim
from src.model.transformer import Transformer
from src.data import get_dataloader
from src import config
import spacy

def download_spacy_models():
    spacy.cli.download("en_core_web_sm")
    spacy.cli.download("de_core_news_sm")

def train():
    # --- Setup ---
    dataloader = get_dataloader()

    model = Transformer(
        n_src_vocab=config.SRC_VOCAB_SIZE,
        n_trg_vocab=config.TRG_VOCAB_SIZE,
        src_pad_idx=config.SRC_PAD_IDX,
        trg_pad_idx=config.TRG_PAD_IDX,
        d_word_vec=config.D_MODEL,
        d_model=config.D_MODEL,
        d_inner=config.D_INNER,
        n_layers=config.N_LAYERS,
        n_head=config.N_HEAD,
        d_k=config.D_K,
        d_v=config.D_V,
        dropout=config.DROPOUT,
        n_position=config.N_POSITION,
        trg_emb_prj_weight_sharing=config.TRG_EMB_PRJ_WEIGHT_SHARING,
        emb_src_trg_weight_sharing=config.EMB_SRC_TRG_WEIGHT_SHARING,
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=config.TRG_PAD_IDX)

    # --- Training ---
    for epoch in range(config.EPOCHS):
        for i, (src_batch, trg_batch) in enumerate(dataloader):
            # This is a placeholder for the actual training step.
            # A real implementation would require converting the text batches
            # to tensors of indices, padding them, and creating masks.
            
            print(f"Epoch: {epoch+1}, Batch: {i+1}")
            print("Source:", src_batch)
            print("Target:", trg_batch)
            print("-" * 20)

            # --- In a real training step ---
            # optimizer.zero_grad()
            # 
            # src_seq = ... # Padded source tensor
            # trg_seq = ... # Padded target tensor
            #
            # outputs = model(src_seq, trg_seq[:, :-1])
            # loss = criterion(outputs.view(-1, outputs.size(-1)), trg_seq[:, 1:].reshape(-1))
            #
            # loss.backward()
            # optimizer.step()

def main():
    download_spacy_models()
    train()

if __name__ == '__main__':
    main()