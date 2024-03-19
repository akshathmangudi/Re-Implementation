import torch
import torch.nn as nn
import torch.utils.tensorboard as SummaryWriter
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset
from config import get_weights


from pathlib import Path
from model import (
    InputEmbeddings,
    PositionalEncoding,
    MHA,
    FeedForward,
    EncoderBlock,
    DecoderBlock,
    Encoder,
    Decoder,
    Projection,
    Transformer,
)


def build(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    layers: int = 6,
    heads: int = 8,
    dropout: float = 0.01,
    d_ff: int = 2048,
):

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    enc_blocks = []
    for _ in range(layers):
        self_att_block = MHA(d_model, heads, dropout)
        feed_forw_block = FeedForward(d_model, d_ff, dropout)
        enc_block = EncoderBlock(self_att_block, feed_forw_block, dropout)
        enc_blocks.append(enc_block)

    dec_blocks = []
    for _ in range(layers):
        self_att_block = MHA(d_model, heads, dropout)
        cross_att_block = MHA(d_model, heads, dropout)
        feed_forw_block = FeedForward(d_model, d_ff, dropout)
        dec_block = DecoderBlock(
            self_att_block, cross_att_block, feed_forw_block, dropout
        )
        dec_blocks.append(dec_block)

    encoder = Encoder(nn.ModuleList(enc_blocks))
    decoder = Decoder(nn.ModuleList(dec_blocks))

    proj_layer = Projection(d_model, tgt_vocab_size)

    # Finally
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer
    )

    # Xavier Initialization:
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


def get_sentences(dataset, lang):
    for data in dataset:
        yield data["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_sentences(ds, lang))
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    tokenizer_src = get_or_build_tokenizer(config, raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, raw, config["lang_tgt"])

    train_ds = len(0.9 * len(raw))
    val_ds = len(raw) - train_ds
    train_raw, val_raw = random_split(raw, [train_ds, val_ds])

    train_ds = BilingualDataset(
        train_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_src = 0
    max_tgt = 0
    for item in raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_src = max(max_src, len(src_ids))
        max_tgt = max(max_tgt, len(tgt_ids))

    print(f"Max length of source: {max_src}")
    print(f"Max length of target: {max_tgt}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.device.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    init_epochs = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights(config, config["preload"])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        init_epochs = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(init_epochs, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            enc_input = batch["encoder_input"].to(device)
            dec_input = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)

            enc_output = model.encode(enc_input, enc_mask)
            dec_output = model.decode(enc_input, enc_mask, dec_input, dec_mask)
            proj_output = model.project(dec_output)

            label = batch["label"].to(device)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():0.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        model_filename = get_weights(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
