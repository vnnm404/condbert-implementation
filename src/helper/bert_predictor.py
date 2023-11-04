import torch

import numpy as np
import copy
import bisect

from torch.utils.data import DataLoader

from keras_preprocessing.sequence import pad_sequences

from helper.utils import bpe_tokenize, nlargest_indexes, remove_masked_token_subwords, merge_sorted_results




class BertPredictor:
    def __init__(
        self,
        model,
        bpe_tokenizer,
        max_len=250,
        mask_in_multiunit=False,
        device=None,
        label=0,
        logits_postprocessor=None,
        contrast_penalty=0,
        mean=np.mean,
        confuse_bert_args=False,
    ):
        self._model = model
        self._bpe_tokenizer = bpe_tokenizer
        self._max_len = max_len
        self._mask_in_multiunit = mask_in_multiunit
        self.device = device or torch.device("cuda")
        self.label = label
        self.logits_postprocessor = logits_postprocessor
        self.contrast_penalty = contrast_penalty
        self.mean = mean
        self.confuse_bert_args = confuse_bert_args

    def generate(
        self,
        b_bpe_tokens,
        b_masked_pos,
        mask_token=True,
        n_top=5,
        n_units=1,
        n_tokens=[1],
        fix_multiunit=True,
        beam_size=10,
        multiunit_lookup=100,
        max_multiunit=10,
        label=None,
    ):
        # Main function to generate predictions for masked tokens, can generate predictions for both single words and sequences.

        # Initialize lists to store prediction results
        result_preds = [[] for _ in range(len(b_bpe_tokens))]
        result_scores = [[] for _ in range(len(b_bpe_tokens))]

        # Generate predictions for single words
        if 1 in n_tokens:
            result_preds, result_scores = self.predict_single_word(
                b_bpe_tokens,
                b_masked_pos,
                mask_token=mask_token,
                n_top=n_top,
                n_units=n_units,
                multiunit_lookup=multiunit_lookup,
                fix_multiunit=fix_multiunit,
                max_multiunit=max_multiunit,
                label=label,
            )

        # Generate predictions for sequences of tokens iterating through n_tokens and predicting that number of tokens
        for n_t in n_tokens:
            if n_t == 1:
                continue

            pred_tokens, pred_scores = self.predict_token_sequence(
                b_bpe_tokens,
                b_masked_pos,
                mask_token=mask_token,
                n_top=n_top,
                n_units=n_units,
                seq_len=n_t,
                multiunit_lookup=multiunit_lookup,
                fix_multiunit=fix_multiunit,
                beam_size=beam_size,
                max_multiunit=max_multiunit,
                label=label,
            )

            # Merge and store the top predictions
            for i in range(len(b_bpe_tokens)):
                result_preds[i], result_scores[i] = merge_sorted_results(
                    result_preds[i],
                    result_scores[i],
                    pred_tokens[i],
                    pred_scores[i],
                    n_top,
                )

        return result_preds, result_scores

    def predict_single_unit(
        self, bpe_tokens, masked_position, mask_token, n_top, label=None
    ):
        """The following function predicts the top tokens for a single masked token position.
        It also tokenizes the input, prepares it for BERT, and obtains predictions for the masked position.
        """

        if label is None:
            label = self.label

        # Make a deep copy of the input tokens
        bpe_tokens = copy.deepcopy(bpe_tokens)

        # Determine the maximum length for token sequences
        max_len = min([max(len(e) for e in bpe_tokens) + 2, self._max_len])
        token_ids = []
        # Add a CLS and SEP token to the start and end along with replacing the toxic word with MASK
        for i in range(len(bpe_tokens)):
            bpe_tokens[i] = bpe_tokens[i][: max_len - 2]

            if mask_token:
                if i >= len(masked_position):
                    continue
                pos = masked_position[i]
                if pos >= len(bpe_tokens[i]):
                    continue
                bpe_tokens[i][pos] = "[MASK]"

            bpe_tokens[i] = ["[CLS]"] + bpe_tokens[i] + ["[SEP]"]

            token_ids.append(self._bpe_tokenizer.convert_tokens_to_ids(bpe_tokens[i]))

        # Pad token sequences, create attention masks and tensors
        token_ids = pad_sequences(
            token_ids, maxlen=max_len, dtype="long", truncating="post", padding="post"
        )
        attention_masks_tensor = torch.tensor(token_ids > 0).long().to(self.device)
        tokens_tensor = torch.tensor(token_ids).to(self.device)

        # Set segment IDs for BERT input
        segments_ids = np.ones_like(token_ids, dtype=int) * label
        segments_tensor = torch.tensor(segments_ids).to(self.device)

        # Set the model in evaluation mode and obtain predictions
        self._model.eval()
        with torch.no_grad():
            if self.confuse_bert_args:
                target_sent = self._model(
                    tokens_tensor,
                    attention_mask=segments_tensor,
                    token_type_ids=attention_masks_tensor,
                )[0]
            else:
                target_sent = self._model(
                    tokens_tensor,
                    token_type_ids=segments_tensor,
                    attention_mask=attention_masks_tensor,
                )[0]

        if self.contrast_penalty:
            with torch.no_grad():
                another = self._model(
                    tokens_tensor,
                    token_type_ids=1 - segments_tensor,
                    attention_mask=attention_masks_tensor,
                )[0]
            diff = torch.softmax(
                target_sent, -1
            ) - self.contrast_penalty * torch.softmax(another, -1)
            target_sent = torch.log(torch.clamp(diff, 1e-20))

        target_sent = target_sent.detach().cpu().numpy()

        # Find the best replacement for the masked toxic word
        final_top_scores = []
        final_top_tokens = []
        for i in range(target_sent.shape[0]):
            row = target_sent[i]
            idx = masked_position[i]
            # Skip if the index is out of bounds
            if idx + 1 >= len(row):
                continue
            logits = row[idx + 1]
            logits = self.adjust_logits(logits, label=label)
            top_ids = nlargest_indexes(logits, n_top)

            # Retrieve top token scores and convert token IDs to tokens
            top_scores = [target_sent[i][masked_position[i] + 1][j] for j in top_ids]
            top_tokens = self._bpe_tokenizer.convert_ids_to_tokens(top_ids)

            # Store the top tokens and their scores
            final_top_scores.append(top_scores)
            final_top_tokens.append(top_tokens)

        return final_top_tokens, final_top_scores

    def adjust_logits(self, logits, label=0):
        if self.logits_postprocessor:
            return self.logits_postprocessor(logits, label=label or 0)
        return logits

    def predict_single_word(
        self,
        bpe_tokens,
        masked_position,
        mask_token,
        n_top,
        n_units,
        fix_multiunit,
        multiunit_lookup,
        max_multiunit,
        label=None,
    ):
        # Predict the top tokens for a single masked word (unit), may involve multiple subword tokens.
        pred_tokens, scores = self.predict_single_unit(
            bpe_tokens, masked_position, mask_token=mask_token, n_top=n_top, label=label
        )

        # Initialize lists to store the top tokens and their scores
        final_pred_tokens = []
        final_scores = []

        for j in range(len(pred_tokens)):
            # If n_units > 1, it considers multiple units for the prediction.
            if n_units > 1:
                pred_tokens[j] = list(reversed(pred_tokens[j][:multiunit_lookup]))
                scores[j] = list(reversed(scores[j][:multiunit_lookup]))

                # Generate multiple unit tokens for the masked position
                seq_list = self.generate_multiunit_token(
                    masked_position[j],
                    bpe_tokens[j],
                    n_top=multiunit_lookup,
                    n_units=n_units,
                    label=label,
                )

                for seq in seq_list[:max_multiunit]:
                    seq_pred, seq_scores = seq
                    multiunit_token = "_".join(seq_pred)

                    # Optionally fix the format of multiunit tokens
                    if fix_multiunit:
                        multiunit_token = multiunit_token.replace("#", "")
                        multiunit_token = multiunit_token.replace("_", "")

                    # Calculate the average score for the multiunit token
                    multiunit_score = self.mean(seq_scores)

                    # Insert the multiunit token and its score into the predictions
                    ind = bisect.bisect(scores[j], multiunit_score)
                    pred_tokens[j].insert(ind, multiunit_token)
                    scores[j].insert(ind, multiunit_score)

                pred_tokens[j] = list(reversed(pred_tokens[j]))
                scores[j] = list(reversed(scores[j]))

            final_pred_tokens.append(pred_tokens[j][:n_top])
            final_scores.append(scores[j][:n_top])

        return final_pred_tokens, final_scores

    def generate_variants(self, bpe_tokens, mask_pos, gen_tokens, gen_scores, seq_len):
        batch_size = len(bpe_tokens)

        if not gen_tokens:
            yield bpe_tokens, [0.0] * batch_size, [
                [] for _ in range(batch_size)
            ], mask_pos
            return

        """Generate variants of input tokens by filling the masked positions with different token sequences.
        Yields these variants along with their scores."""

        for var_num in range(len(gen_tokens[0])):
            # Skip empty variants
            if not gen_tokens[0][var_num]:
                continue

            variant = []
            new_mask = []
            var_t = []
            var_s = []

            # Process each row of input tokens
            for i in range(batch_size):
                new_bpe = copy.deepcopy(bpe_tokens[i])

                # Fill masked positions with generated tokens
                for seq_num in range(len(gen_tokens[i][var_num])):
                    new_bpe[mask_pos[i] + seq_num] = gen_tokens[i][var_num][seq_num]

                var_t.append(gen_tokens[i][var_num])
                var_s.append(gen_scores[i][var_num])

                # Update the new masked positions
                new_mask.append(mask_pos[i] + len(gen_tokens[i][var_num]))

                variant.append(new_bpe)

            yield variant, var_s, var_t, new_mask

    def update_beam(
        self, prev_tokens, prev_score, new_scores, new_tokens, gen_scores, gen_tokens
    ):
        # Update the beam of predictions. Keep track of the top predictions based on their scores.

        for i in range(len(gen_scores)):
            # Calculate the final score for the generated prediction
            final_gen_score = prev_score + gen_scores[i]

            # Find the position to insert the new prediction in the beam
            insert_pos = bisect.bisect(new_scores, final_gen_score)

            new_scores.insert(insert_pos, final_gen_score)
            del new_scores[0]

            new_tokens.insert(insert_pos, prev_tokens + [gen_tokens[i]])

            # Remove the oldest prediction if the beam size is exceeded
            if len(new_tokens) > len(new_scores):
                del new_tokens[0]

    def predict_token_sequence(
        self,
        bpe_tokens,
        masked_pos,
        mask_token,
        n_top,
        seq_len,
        beam_size,
        n_units,
        fix_multiunit,
        multiunit_lookup,
        max_multiunit,
        label=None,
    ):
        bpe_tokens = copy.deepcopy(bpe_tokens)

        batch_size = len(bpe_tokens)
        for i in range(batch_size):
            for seq_num in range(seq_len - 1):
                bpe_tokens[i].insert(masked_pos[i] + 1, "[MASK]")

        # Predict a sequence of tokens for a masked position using beam search.
        gen_scores = []
        gen_tokens = []
        for seq_num in range(seq_len):
            gen_scores_seq = [
                [0.0 for __ in range(beam_size)] for _ in range(batch_size)
            ]
            gen_tokens_seq = [
                [[] for __ in range(beam_size)] for _ in range(batch_size)
            ]
            for variant, variant_score, prev_tokens, new_mask in self.generate_variants(
                bpe_tokens, masked_pos, gen_tokens, gen_scores, seq_len=seq_len
            ):
                top_tokens, top_scores = self.predict_single_word(
                    variant,
                    new_mask,
                    mask_token=True,
                    n_top=n_top,
                    n_units=n_units,
                    fix_multiunit=fix_multiunit,
                    multiunit_lookup=multiunit_lookup,
                    max_multiunit=max_multiunit,
                    label=label,
                )

                for i in range(batch_size):
                    self.update_beam(
                        prev_tokens[i],
                        variant_score[i],
                        gen_scores_seq[i],
                        gen_tokens_seq[i],
                        top_scores[i],
                        top_tokens[i],
                    )

            gen_tokens = gen_tokens_seq
            gen_scores = gen_scores_seq

        # Calculate the average scores for generated tokens
        gen_scores = [[(e / seq_len) for e in l] for l in gen_scores]

        return (
            [list(reversed(e)) for e in gen_tokens],
            [list(reversed(e)) for e in gen_scores],
        )
