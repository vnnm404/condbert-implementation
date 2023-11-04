import torch
from collections import defaultdict
from helper.utils import group_by_first_token, tokens_to_string


class CondBert:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        neg_words,
        pos_words,
        word2coef,
        token_toxicities,
        predictor=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neg_words = neg_words
        self.pos_words = pos_words
        self.word2coef = word2coef
        self.token_toxicities = token_toxicities
        self.predictor = predictor

        # calculated properties
        self.v = {v: k for k, v in tokenizer.vocab.items()}
        self.device_toxicities = torch.tensor(token_toxicities).to(self.device)

        self.neg_complex_tokens = group_by_first_token(neg_words, self.tokenizer)
        self.pos_complex_tokens = group_by_first_token(pos_words, self.tokenizer)
        self.mask_index = self.tokenizer.convert_tokens_to_ids("[MASK]")

    
    def mask_toxic_tokens(self, sentence, bad_words, masks, sent_id, aggressive, max_score_margin, min_bad_score):
        len_sentence = len(sentence)
        token_id = 0

        while token_id < len_sentence:
            token = sentence[token_id]
            toxic_hypotheses = bad_words.get(token, [])

            for hypothesis in toxic_hypotheses:
                hypothesis_length = len(hypothesis)
                end_index = token_id + hypothesis_length

                if sentence[token_id:end_index] == hypothesis:
                    masks[sent_id, token_id:end_index] = 1

                    for suffix_id, next_token in enumerate(sentence[end_index:], start=end_index):
                        if self.tokenizer.convert_ids_to_tokens(next_token).startswith("##"):
                            masks[sent_id, suffix_id] = 1
                        else:
                            break

                    token_id = end_index - 1
                    break
            token_id += 1

        return masks

    def score_and_mask_words(self, sentence, masks, sent_id, max_score_margin, min_bad_score):
        scored_words = [
            (indices, word, score)
            for indices, word in tokens_to_string(sentence,self.v)
            if (score := self.word2coef.get(word, 0))
        ]

        if scored_words:
            max_score = max(score for _, _, score in scored_words)
            if max_score > min_bad_score:
                for indices, word, score in scored_words:
                    if score >= max(min_bad_score, max_score * max_score_margin):
                        masks[sent_id, indices] = 1

        return masks

    def get_mask_fast(self, inp: str, aggressive=True, max_score_margin=0.5, min_bad_score=0.0):
        bad_words = self.neg_complex_tokens
        sentences = [self.tokenizer.encode(inp, add_special_tokens=True)]
        sentences_torch = torch.tensor(sentences)
        masks = torch.zeros_like(sentences_torch)

        for sent_id, sentence in enumerate(sentences):
            masks = self.mask_toxic_tokens(sentence, bad_words, masks, sent_id, aggressive, max_score_margin, min_bad_score)

            if torch.sum(masks[sent_id]).item() == 0 or aggressive:
                masks = self.score_and_mask_words(sentence, masks, sent_id, max_score_margin, min_bad_score)

        return sentences_torch, masks


    def convert_mask(self, tok_ids, mask_ids, duplicate=False, start_from=0):
        # We find the first masked word, keep only its first token and get its position
        toks_tmp = [self.tokenizer.convert_ids_to_tokens(tok_ids[0])[1:-1]]
        mask_pos = None
        toks = []
        mask_toks = []
        has_mask = False
        for i, is_masked in enumerate(mask_ids[0][1:-1]):
            tok = toks_tmp[0][i]
            if not has_mask:
                if is_masked and i >= start_from and not tok.startswith("##"):
                    has_mask = True
                    mask_pos = [i]
                    mask_toks.append(tok)
                toks.append(tok)
            else:
                if not is_masked or not tok.startswith("##"):
                    toks.extend(toks_tmp[0][i:])
                    break
                else:
                    mask_toks.append(tok)
        toks = [toks]

        if duplicate:
            toks = [toks_tmp[0] + ["[SEP]"] + toks[0]]
            mask_pos[0] += len(toks_tmp[0]) + 1
        return toks, mask_pos, mask_toks

    def replacement_loop(
        self,
        text,
        span_detector=None,
        predictor=None,
        verbose=True,
        chooser=None,
        n_tokens=(1, 2, 3),
        n_top=10,
        mask_token=False,
        max_steps=1000,
        label=0,
    ):
        # Perform a replacement loop to detoxify text.
        if span_detector is None:
            span_detector = self.get_mask_fast
        if predictor is None:
            predictor = self.predictor
        new_text = text
        look_from = 0

        for _ in range(max_steps):
            tok_ids, mask_ids = span_detector(new_text)
            # If the sentence doesnt contain any toxic words, return as is
            if not sum(mask_ids[0][(1 + look_from) :]):
                break
            # Convert masked tokens into replacement candidates
            toks, mask_pos, mask_toks = self.convert_mask(
                tok_ids, mask_ids, duplicate=False, start_from=look_from
            )

            # If there is no mask, and thus no toxic word, return as is
            if mask_pos is None:
                return new_text

            # Generate replacement candidates for the toxic tokens
            texts, scores = predictor.generate(
                toks,
                mask_pos,
                n_tokens=list(n_tokens),
                n_top=n_top,
                mask_token=mask_token,
                label=label,
            )

            # Choose the best replacement hypothesis
            old_replacement = chooser(
                hypotheses=texts[0], scores=scores[0], original=mask_toks
            )

            if isinstance(old_replacement, str):
                old_replacement = [old_replacement]

                # Split the replacement into tokens
            replacement = [t for w in old_replacement for t in w.split("_")]

            if verbose:
                print(mask_toks, "->", replacement)

            # Update the text with the replacement
            new_toks = toks[0][: mask_pos[0]] + replacement + toks[0][mask_pos[0] + 1 :]
            new_text = self.tokenizer.convert_tokens_to_string(new_toks)
            look_from = mask_pos[0] + len(old_replacement)

        return new_text
