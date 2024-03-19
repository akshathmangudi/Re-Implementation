import torch
from torch.utils.data import Dataset
from typing import Any


class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.eos_token = torch.Tensor(
            [tokenizer_src.token_to_id(["[EOS]"])], dtype=torch.int64
        )
        self.sos_token = torch.Tensor(
            [tokenizer_src.token_to_id(["[SOS]"])], dtype=torch.int64
        )
        self.pad_token = torch.Tensor(
            [tokenizer_src.token_to_id(["[PAD]"])], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_pair = self.ds[index]
        src_text = src_pair["translation"][src_lang]
        tgt_text = src_pair["translation"][tgt_lang]

        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_src.decode(tgt_text).ids

        enc_padding = self.seq_len - len(enc_tokens) - 2
        dec_padding = self.seq_len - len(dec_tokens) - 1

        if enc_padding < 0 or dec_padding < 0:
            raise ValueError("Too long")

        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_padding, type=torch.int64),
            ]
        )

        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_padding, type=torch.int64),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input, type=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_padding, type=torch.int64),
            ]
        )

        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "encoder_mask": (enc_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (dec_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(dec_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
