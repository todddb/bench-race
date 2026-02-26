# monkeypatch_tokenizer.py
# Add a compatibility accessor `all_special_tokens_extended` to HF tokenizers that
# are missing it so vLLM's calls won't crash. Best-effort only.
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
except Exception:
    PreTrainedTokenizerBase = None

if PreTrainedTokenizerBase is not None and not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    def _get_all_special_tokens_extended(self):
        out = []
        try:
            if hasattr(self, "all_special_tokens") and self.all_special_tokens:
                out = list(self.all_special_tokens)
            elif hasattr(self, "additional_special_tokens") and self.additional_special_tokens:
                out = list(self.additional_special_tokens)
            else:
                for k in ("bos_token", "eos_token", "unk_token", "pad_token", "cls_token", "sep_token"):
                    v = getattr(self, k, None)
                    if v:
                        out.append(v)
        except Exception:
            out = []
        return out
    try:
        PreTrainedTokenizerBase.all_special_tokens_extended = property(_get_all_special_tokens_extended)
    except Exception:
        pass
