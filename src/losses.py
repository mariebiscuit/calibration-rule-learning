import numpy as np
import os
import torch
from tqdm import tqdm
import pickle
import copy
from typing import Type, Dict, List

"""
=== Experiment 3 ===
Different fine-tuning losses
"""

def __get_target_idx(
    tokenized_paragraph: Type[torch.Tensor],
    tokenizer,
    start_symbols: str = [" ->"],
    end_symbols=["\n", "\n\n"],
):
    """
    Get idxes of answer tokens for one tokenized paragraph, where
    answer token(s) are bounded by one of the start symbols and one of the end symbols
    """
    start_symbol_ids = [
        tokenizer.encode(start_symbol, add_special_tokens=False)
        for start_symbol in start_symbols
    ]
    end_symbol_ids = [
        tokenizer.encode(end_symbol, add_special_tokens=False)
        for end_symbol in end_symbols
    ]
    start_idxes = []
    for start_symbol_id in start_symbol_ids:
        if len(start_symbol_id) == 1:
            start_idxes.append(
                np.arange(len(tokenized_paragraph))[
                    np.array(tokenized_paragraph) == start_symbol_id
                ]
            )
        else:  # Handle multi-token start symbols
            idxes_for_symbol = np.arange(len(tokenized_paragraph))[
                np.array(tokenized_paragraph) == start_symbol_id[-1]
            ]  # find occurences of the last token in the start symbol

            # For each preceding token in the start symbol, progressively slide window backwards to check that tokens match the preceding start symbol tokens
            for i, symbol_part in enumerate(start_symbol_id[::-1][1:]):
                taken_indexes = (
                    tokenized_paragraph[idxes_for_symbol - (i + 1)] == symbol_part
                )
                if len(taken_indexes) == 1:
                    taken_indexes = (
                        taken_indexes.item()
                    )  # need to take out singleton list for some reason

                idxes_for_symbol = idxes_for_symbol[
                    np.arange(len(idxes_for_symbol))[taken_indexes]
                ]

            start_idxes.append(idxes_for_symbol)

    start_idxes = np.sort(np.concatenate(start_idxes))

    #  Handle contiguous start symbols: we take the last start_symbol:
    # [3, 4, 5, 6, 9 ,10, 11, 15, 19] -> [6, 11, 15, 19]
    start_idxes = start_idxes[
        np.append(
            np.arange(len(start_idxes) - 1)[(start_idxes[:-1] - start_idxes[1:]) != -1],
            len(start_idxes) - 1,
        )
    ]

    end_idxes = []
    for end_symbol_id in end_symbol_ids:
        if len(end_symbol_id) == 1:
            idxes_for_symbol = np.arange(len(tokenized_paragraph))[
                np.array(tokenized_paragraph) == end_symbol_id
            ]
        else:  # Handle multi-token end symbols
            idxes_for_symbol = np.arange(len(tokenized_paragraph))[
                np.array(tokenized_paragraph) == end_symbol_id[0]
            ]  # find occurences of the last token in the start symbol

            # for each preceding token in the start symbol, progressively slide window forwards 
            # to check that tokens match the following end symbol tokens
            for i, symbol_part in enumerate(end_symbol_id[1:]):
                taken_indexes = (
                    tokenized_paragraph[idxes_for_symbol + (i + 1)] == symbol_part
                )
                if len(taken_indexes) == 1:
                    taken_indexes = (
                        taken_indexes.item()
                    )  # need to take out singleton list for some reason
                idxes_for_symbol = idxes_for_symbol[
                    np.arange(len(idxes_for_symbol))[taken_indexes]
                ]

        idxes_for_symbol = idxes_for_symbol[
            idxes_for_symbol > start_idxes[0]
        ]  # ignore end_symbols that appeared before the first start_symbol
        end_idxes.append(idxes_for_symbol)

    end_idxes = np.sort(np.concatenate(end_idxes))
    #  Handle contiguous end symbols: we take the first end symbol:
    # [3, 4, 5, 6, 9 ,10, 11, 15, 19] ->  [ 3,  9, 15, 19]
    end_idxes = end_idxes[
        np.insert(
            np.arange(1, len(end_idxes))[(end_idxes[1:] - end_idxes[:-1]) != 1], 0, 0
        )
    ]

    # Take the first end_idxes that correspond to a start_idx
    end_idxes_final = []
    for start_idx in start_idxes:
        end_idxes_final.append(end_idxes[np.argmax(end_idxes > start_idx)])
    end_idxes = end_idxes_final

    assert len(end_idxes) == len(
        start_idxes
    ), f"Num of start {len(start_idxes)} and end idxes {len(end_idxes)} are not equal"

    answer_idxes = []
    for start, end in zip(start_idxes, end_idxes_final):
        answer_idxes += list(
            range(start + 1, end)
        )  # adds start+1 token idx until end-1 token idx: START [ tok tok tok ] END

    return answer_idxes


def get_target_idxes(
    tokenizer,
    input_ids: Type[torch.Tensor],
    return_padded_tensor: bool,
    for_causal_modeling: bool = True,
):
    """
    Get idxes of answer tokens for every tokenized paragraph in the dataset, so that
    answer tokens can be picked out of seq_len length tensor

    :param torch.Tensor input_ids: tokenized tensor
    :param bool return_padded_tensor: if True, will pad out all idxes with the last idx of each entry
    :param bool for_causal_modeling: if True, will minus 1 from the actual answer tokens because it is the previous word that predicts the next word
    """
    causal_offset = int(for_causal_modeling)
    target = [
        (torch.Tensor(__get_target_idx(i, tokenizer)) - causal_offset).to(torch.int64) for i in input_ids
    ]
    
    if return_padded_tensor:
        max_rows = max([len(row) for row in target])
        # print([len(row) for row in target])
        padded = [torch.cat((x, torch.tile(x[-1], (max_rows-len(x), ))), axis=0) for x in target]
        # padded = [x.expand(max_rows) for x in target]
        if len(padded) == 1:
            output = padded[0]
        else:
            output = torch.vstack(padded)
    else:
        output = target

    return output


def compute_distribution_loss(
    model, tokenizer, inputs, true_tok, false_tok):
    """
    Loss as KL-Divergence between humam proportions of True/False as soft labels and models'
    probability distribution over vocabulary

    :returns:
        - pred_dist: Prob distribution over tokens at answer tokens
        - kl: KL-divergence as loss
    """

    ids = inputs['input_ids'] if isinstance(inputs['input_ids'], list) else inputs['input_ids'].cpu()
    ids = [ids] if isinstance(ids[0], int) else ids

    keep_seq_idx = get_target_idxes(
            tokenizer,
            ids,
            return_padded_tensor=False, # !! this must be False or the rest of the code wil break...
            for_causal_modeling=True)
            # keep_seq_idx here is a List[torch.Tensor]

    indexes = inputs.pop("idx")
    hdists = inputs.pop("hdist")

    if torch.Tensor(inputs['input_ids']).dim() == 1:
        input_ids = torch.Tensor(inputs['input_ids']).to(int).reshape(1, -1)
        attn_mask = torch.Tensor(inputs['attention_mask']).to(int).reshape(1, -1)
    else:
        input_ids = torch.Tensor(inputs['input_ids']).to(int)
        attn_mask = torch.Tensor(inputs['attention_mask']).to(int)
        
    outputs = model(input_ids=input_ids,
                    attention_mask=attn_mask)

    logits = outputs["logits"]  # batch x seq_len x embedding
    batch_size, seq_len, emb_size = logits.shape
    device = outputs["logits"].device
    
    hdist = torch.cat([torch.Tensor(h) for h in hdists], axis=0).to(device)  # (batch x num_ans) x 2
    if hdist.dim() != 2 or hdist.shape[1] != 2:
        hdist = hdist.reshape((-1, 2))


    probs = torch.nn.functional.softmax(
        logits, dim=2
    )  # softmax across vocab, (batch x seq_len x emb_size)

    if (len(keep_seq_idx) > 0):  # [[num_ans for concept 1], [num_ans for concept 2]...]
        for i in range(1, len(keep_seq_idx)):
            keep_seq_idx[i] = keep_seq_idx[i] + (
                i * seq_len
            )  # flattened while accounting for batch
        keep_seq_idx = torch.cat(keep_seq_idx, axis=0).to(torch.int64).to(device)
    else: # just [[num_ans for concept 1]]
        keep_seq_idx = torch.Tensor(keep_seq_idx[0]).to(torch.int64).to(device)

    probs = probs.reshape(-1, emb_size)  # flattened to (batch*seq_len) x emb_size

    pred_dist = probs[keep_seq_idx] + (
        torch.e**-100
    )  # take only prob dist of answer tokens in sequence, smooth out 0s
        # should be num_answers * emb_sz

    # build vocab-size distribution with only values at True and False token
    hlike_dist = torch.zeros_like(pred_dist).to(device) # num_answers * emb_sz

    if hdist[:, 0].dim() < hlike_dist[:, true_tok].dim() :
        hlike_dist[:, true_tok] = hdist[:, 0].reshape((-1, 1))
        hlike_dist[:, false_tok] = hdist[:, 1].reshape((-1, 1))
    else:       
        hlike_dist[:, true_tok] = hdist[:, 0]
        hlike_dist[:, false_tok] = hdist[:, 1]

    kl = torch.nn.functional.kl_div(
        torch.log(pred_dist), hlike_dist, reduction="batchmean"
    )

    return pred_dist, kl


def compute_restricted_ce_loss(model, tokenizer, inputs):
    """
    LM cross-entropy loss but only at answer tokens
    """
    ids = inputs['input_ids'] if isinstance(inputs['input_ids'], list) else inputs['input_ids'].cpu()

    keep_seq_idx = get_target_idxes(
            tokenizer,
            ids,
            return_padded_tensor=True,
            for_causal_modeling=True).to(torch.int64)
    # batch x num answers (2 x 546)
    # inner list contains indexes to answer tokens in within seq_len tokens (e.g. [[175, 176, 177...]])

    outputs = model(input_ids=torch.Tensor(inputs['input_ids']).to(int), 
                    attention_mask=torch.Tensor(inputs['attention_mask']).to(int))
    logits = outputs["logits"]  # batch x seq_len x embedding
    batch_size, seq_len, emb_size = logits.shape

    device = outputs["logits"].device
    keep_seq_idx = keep_seq_idx.to(device)

    probs = torch.nn.functional.softmax(
        logits, dim=2
    )  # softmax across vocab, (batch x seq_len x emb_size)

    input_ids = inputs["input_ids"]  # (batch x seq len)
    labels = torch.cat(
        (input_ids[:, 1:], torch.ones((batch_size, 1)).to(device)), -1
    ).to(
        torch.int64
    )  # (batch x seq len)
    
    if len(keep_seq_idx.shape) == 1:
        keep_seq_idx = torch.unsqueeze(keep_seq_idx, 0)
        
    only_answers = torch.gather(labels, 1, keep_seq_idx).to(
        device
    )  # get token_ids of only answer tokens
    new_labels = (
        (torch.ones_like(labels) * -100)
        .to(device)
        .scatter_(1, keep_seq_idx, only_answers)
        .to(torch.int64)
    )
    # batch x seq len, [-100, -100, .... 175, 176, 177] i.e. only target tokens, -100 everyhwhere else

    probs = probs.reshape((-1, probs.shape[2]))  # flatten batches
    new_labels = new_labels.reshape(-1).to(torch.int64)  # flatten batches

    ce = torch.nn.functional.cross_entropy(
        probs, new_labels, reduction="mean", ignore_index=-100
    )
    return probs, ce
