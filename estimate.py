#!/usr/bin/env python
# coding: utf-8

# estimates the tv distance between two models given as cmdline input


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, GenerationConfig
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AdamW, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, TopPLogitsWarper
import locale
import gmpy2
from gmpy2 import mpfr
import random, csv
from math import log as _log, ceil
from rich.progress import track
import json
from copy import copy
from accelerate import Accelerator
accelerator = Accelerator()


locale.getpreferredencoding = lambda: "UTF-8"

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
gmpy2.set_context(gmpy2.context())

models_folder = "trained_models"
if not os.path.exists(models_folder):
   os.mkdir(models_folder)


def _prsample(prompt, model, tokenizer, gen_config, stopping_criteria, seed):
    r"""
     _prsample (probablity revealing sample) returns a sampled sentence (token sequence) along 
     with its generation probability. 
    """
    set_seed(seed)
    
    encoded_input = tokenizer(prompt, return_tensors='pt').to(torch_device)
    
    outputs = model.generate(**encoded_input, generation_config=gen_config, stopping_criteria = stopping_criteria, pad_token_id=tokenizer.eos_token_id)

    gen_sequences = outputs.sequences[:, encoded_input.input_ids.shape[-1]:]
    # gen_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    gen_text = []
    decodedtext = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)
    for text in decodedtext:
        gen_text.append(prompt + text) 
        # todo : is it pefect?
    
    i = 0; prob = [1] * gen_config.num_return_sequences
    condProbs = [[] for i in range(gen_config.num_return_sequences)]

    # outputs.scores : tuple of generation length
    # gen_sequences : tensor of num_return_sequence x generation length
    # probs : num_return_sequences many distributions

    for i in range(len(outputs.scores)):
        probs = nn.functional.softmax(outputs.scores[i], dim=-1)
        assert len(probs) == gen_config.num_return_sequences
        for k in range(gen_config.num_return_sequences): 
            tok = gen_sequences[k][i]
            if tok == model.config.eos_token_id:
                prob[k] *= mpfr(1)
                condProbs[k].append(mpfr(1))
            else:
                prob[k] *= mpfr(probs[k][tok].item())
                condProbs[k].append(mpfr(probs[k][tok].item()))
            
    return gen_text, prob, outputs.sequences, condProbs


def _eval_no_logits(prompt, model, tokenizer, encoded_text, generation_config, seed):
    r"""
     _eval_no_logits (evaluate with no logit access) returns the generation probability of a sentence 
     (token sequence) in a model. This version of _eval assumes no logits access to the model.
    """
    set_seed(seed)

    # tokenize input prompt
    encoded_prompt = tokenizer(prompt, return_tensors='pt').to(torch_device)
    encoded_text = encoded_text[0][len(encoded_prompt.input_ids[0]):]
    
    probarray, probarrayraw = [], []
    prob = 1
    
    for tok in encoded_text:

        # generate next token distribution
        output = model.generate(**encoded_prompt, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
        probs = torch.stack(output.scores, dim=1).softmax(-1)
        
        # eval sentence probability
        probarrayraw.append(probs[0][0][tok].item())
        probarray.append(mpfr(probs[0][0][tok].item()))
        prob *= mpfr(probs[0][0][tok].item())
        
        tensor_buff = torch.cat([encoded_prompt.input_ids, torch.tensor([[tok]]).to(torch_device)], dim = -1)
        encoded_prompt['input_ids'] = tensor_buff
        tensor_buff = torch.cat((encoded_prompt.attention_mask[0], torch.tensor([1]).to(torch_device)), dim = 0)
        encoded_prompt['attention_mask'] = tensor_buff.unsqueeze(0)
    
    #print("Eval (raw): ", probarrayraw)
    
    return prob, probarray


def _expand_dict_for_generation(dict_to_expand, expand_size):
    for key in dict_to_expand:
        if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
            dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
    return dict_to_expand


def _eval(prompt, model, tokenizer, texts, generation_config, collision_list):
    r"""
     _eval (evaluate) returns the generation probability of a sentence (token sequence) in a model . 
     this version of _eval assumes logits access to the model.
    """
    # set_seed(seed)

    tokenizer.pad_token = tokenizer.eos_token 

    ntext = len(texts)

    # tokenize input prompt
    encoded_prompt = tokenizer(prompt, return_tensors='pt').to(torch_device)
    encoded_prompt['past_key_values'] = None
    _prompt_len = encoded_prompt['input_ids'].shape[-1]
    encoded_prompt['position_ids'] = torch.tensor([list(range(_prompt_len))]).to(torch_device) 
    
    probarray, probarrayraw = [], []
    prob = [1] * generation_config.num_return_sequences
    probarray = [[] for i in range(generation_config.num_return_sequences)]
    _prompt_len -= 1 
    logits_processor = LogitsProcessorList()
    logits_processor.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

    encoded_text = tokenizer(texts, return_tensors='pt', padding=True).to(torch_device)
    # generated part of text  
    for key in encoded_text.keys():    
        encoded_text[key] = encoded_text[key][:, encoded_prompt[key].shape[-1]:]
    encoded_prompt = _expand_dict_for_generation(encoded_prompt, generation_config.num_return_sequences)

        
    # for tok in encoded_text:
    
    for i in range(encoded_text.input_ids.shape[-1]):
        
        toks = encoded_text.input_ids[:, i:i+1]
        which_finished = (toks == model.config.eos_token_id)
        which_unfinished = ~which_finished
        # generate next token distribution
        output = model(**encoded_prompt, return_dict=True)
        next_token_logits = output.logits[:, -1, :]
        next_token_scores = logits_processor(encoded_prompt.input_ids, next_token_logits)
        #next_token_scores = logits_warper(encoded_prompt.input_ids, next_token_scores) 
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        
        # eval sentence probability
        # probarrayraw.append(probs[0][tok].item())
        # probarray.append(mpfr(probs[0][tok].item()))
        for at in range(len(prob)):
            check = mpfr(probs[at][toks[at].item()].item())
            # if check == 0: check = 1
            prob[at] *= (check * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())
            probarray[at].append(check * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())
        # print(prob, toks, which_finished)

        input_ids = toks.to(torch_device)
        encoded_prompt['input_ids'] = input_ids

        next_attentions = torch.tensor([[1]] * encoded_prompt.attention_mask.shape[0]).to(torch_device) *  which_unfinished
        attention_mask = torch.cat((encoded_prompt.attention_mask, next_attentions), dim = -1) # todo: unfinished sequences check
        encoded_prompt['attention_mask'] = attention_mask #.unsqueeze(0)
        
        past_key_values = output.past_key_values
        encoded_prompt['past_key_values'] = past_key_values

        _prompt_len += 1
        encoded_prompt['position_ids'] = torch.tensor([[_prompt_len]] * encoded_prompt.attention_mask.shape[0]).to(torch_device)

        # if tok == model.config.eos_token_id:
        #     break
     
    
    #print("Eval (raw): ", probarrayraw)
    
    return prob, probarray, encoded_text.input_ids


def _evalwithchoices(prompt, model, tokenizer, texts, generation_config, collision_list):
   # sourcery skip: remove-dict-keys
    r"""
        _eval (evaluate) returns the generation probability of a sentence (token sequence) in a model . 
        this version of _eval assumes logits access to the model.
        """
    # set_seed(seed)

    tokenizer.pad_token = tokenizer.eos_token 
    
    # tokenize input prompt
    encoded_prompt = tokenizer(prompt, return_tensors='pt').to(torch_device)
    encoded_prompt['past_key_values'] = None
    _prompt_len = encoded_prompt['input_ids'].shape[-1]
    encoded_prompt['position_ids'] = torch.tensor([list(range(_prompt_len))]).to(torch_device) 

    probarray, probarrayraw = [], []
    prob = [1] * generation_config.num_return_sequences
    probarray = [[] for _ in range(generation_config.num_return_sequences)]
    _prompt_len -= 1
    logits_processor = LogitsProcessorList()
    logits_processor.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

    encoded_text = tokenizer(texts, return_tensors='pt', padding=True).to(torch_device)
    # generated part of text  
    for key in encoded_text.keys():    
        encoded_text[key] = encoded_text[key][:, encoded_prompt[key].shape[-1]:]
    encoded_prompt = _expand_dict_for_generation(encoded_prompt, generation_config.num_return_sequences)

    # convert token sequence to string for comparison
    Y=tuple(encoded_text['input_ids'].tolist()[-1])
    checkpos = {}
    for subs in collision_list.keys():
        X = tuple(subs)
        positions = [i for i in range(len(Y) - len(X) + 1) if Y[i:i + len(X)] == X]
        for position in positions:
            checkpos[position] = subs
    
    # this version only supports num_return_sequences = 0
    assert len(prob) == 1
    prob1 = prob #prob1 is the current tokenization prob
    prob2 = copy(prob)
    encoded_prompt1 = encoded_prompt
    encoded_prompt2 = copy(encoded_prompt)
    _prompt_len1 = _prompt_len2 = _prompt_len
    c1 = c2 = 0 # ptrs to two possible paths c1 -> current tokenization path, c2 -> high prob path,
    enc1, enc2 = [], []
    diverging = False

    for _ in range(encoded_text.input_ids.shape[-1]):

        if c1 >= encoded_text.input_ids.shape[-1]: break
        
        if diverging and c1 == c2:
            diverging = False
            if prob1[0] < prob2[0]:
                encoded_prompt1 = copy(encoded_prompt2)
                _prompt_len1 = _prompt_len2
                prob1 = copy(prob2)
                enc1 = copy(enc2)
            else:
                encoded_prompt2 = copy(encoded_prompt1)
                _prompt_len2 = _prompt_len1
                prob2 = copy(prob1)
                enc2 = copy(enc1)

        toks = encoded_text.input_ids[:, c1:c1+1]
        which_finished = (toks == model.config.eos_token_id)
        which_unfinished = ~which_finished
        
        # generate next token distribution
        output = model(**encoded_prompt1, return_dict=True)
        next_token_logits = output.logits[:, -1, :]
        next_token_scores = logits_processor(encoded_prompt1.input_ids, next_token_logits)
        #next_token_scores = logits_warper(encoded_prompt.input_ids, next_token_scores) 
        probs = nn.functional.softmax(next_token_scores, dim=-1)

        
        if c1 in checkpos:
            diverging = True
            othertoks = torch.tensor([[collision_list[checkpos[c1]]]])
            
            for at in range(len(prob)):
                check1 = mpfr(probs[at][toks[at].item()].item())
                check2 = mpfr(probs[at][othertoks[at].item()].item())
                # if check == 0: check = 1
                prob1[at] *= (check1 * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())
                prob2[at] *= (check2 * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())
                probarray[at].append(check1 * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())

            input_ids = toks.to(torch_device)
            encoded_prompt1['input_ids'] = input_ids
            input_ids = othertoks.to(torch_device)
            encoded_prompt2['input_ids'] = input_ids
            
            next_attentions = torch.tensor([[1]] * encoded_prompt1.attention_mask.shape[0]).to(torch_device) *  which_unfinished
            attention_mask = torch.cat((encoded_prompt1.attention_mask, next_attentions), dim = -1) # todo: unfinished sequences check
            encoded_prompt1['attention_mask'] = attention_mask #.unsqueeze(0)
            past_key_values = output.past_key_values
            encoded_prompt1['past_key_values'] = past_key_values
            _prompt_len1 += 1
            encoded_prompt1['position_ids'] = torch.tensor([[_prompt_len1]] * encoded_prompt1.attention_mask.shape[0]).to(torch_device)
            
            next_attentions = torch.tensor([[1]] * encoded_prompt2.attention_mask.shape[0]).to(torch_device) *  which_unfinished
            attention_mask = torch.cat((encoded_prompt2.attention_mask, next_attentions), dim = -1) # todo: unfinished sequences check
            encoded_prompt2['attention_mask'] = attention_mask #.unsqueeze(0)
            past_key_values = output.past_key_values
            encoded_prompt2['past_key_values'] = past_key_values
            _prompt_len2 += 1
            encoded_prompt2['position_ids'] = torch.tensor([[_prompt_len2]] * encoded_prompt2.attention_mask.shape[0]).to(torch_device)

            enc1.append(toks[0][0])
            enc2.append(othertoks[0][0])
            c2 += len(checkpos[c1])
            c1 += 1
                    
        else:

            for at in range(len(prob)):
                check1 = mpfr(probs[at][toks[at].item()].item())
                prob1[at] *= (check1 * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())
                probarray[at].append(check1 * which_unfinished[at].item() + mpfr(1) * which_finished[at].item())

            input_ids = toks.to(torch_device)
            encoded_prompt1['input_ids'] = input_ids
            
            next_attentions = torch.tensor([[1]] * encoded_prompt1.attention_mask.shape[0]).to(torch_device) *  which_unfinished
            attention_mask = torch.cat((encoded_prompt1.attention_mask, next_attentions), dim = -1) # todo: unfinished sequences check
            encoded_prompt1['attention_mask'] = attention_mask #.unsqueeze(0)
            past_key_values = output.past_key_values
            encoded_prompt1['past_key_values'] = past_key_values
            _prompt_len1 += 1
            encoded_prompt1['position_ids'] = torch.tensor([[_prompt_len1]] * encoded_prompt1.attention_mask.shape[0]).to(torch_device)

            enc1.append(toks[0][0])
            c1 += 1

            if not diverging:
                prob2 = copy(prob1)
                encoded_prompt2 = copy(encoded_prompt1)
                c2 = c1
                _prompt_len2 += 1
                enc2 = copy(enc1)
                
    return prob1, probarray, torch.tensor([enc1]).to(torch_device) #encoded_text.input_ids


def estimateTV(model1 = None, model2 = None, tokenizer1 = None, tokenizer2 = None, prompt = "Hi", numsamples = None, dumpFile = "None.txt", batch_size = 1, verbose = 0):
    '''
    estimates TV distance betwene model1 and model2
    '''

    # parameter setup
    seed = 100
    eps = 0.1
    delta = 0.1

    dp = open(dumpFile, "w")

    if numsamples is None:
        m = int(2 / eps**2 * _log(4 / delta)) + 1
    else:
        m = numsamples

    # fix generation length
    ndim = 100 

    # gen config for _prsample
    gen_config1 = GenerationConfig(
        top_k=0,
        top_p=1,
        do_sample=True,
        #early_stopping=True,       #used for beam based method
        max_new_tokens = ndim,           # fix the generation length
        decoder_start_token_id=0,
        eos_token_id=model1.config.eos_token_id,
        pad_token_id=model1.config.eos_token_id,
        return_dict_in_generate=True,    # if commented the outputs of model.generate will be the sequence only
        output_scores=True,
        num_return_sequences=batch_size,
    )


    # genconfig for _eval
    gen_config2 = GenerationConfig(
        top_k=0,
        top_p=1,
        do_sample=True,
        #early_stopping=True,       #used for beambased method
        max_new_tokens = 1,           # fix the generation length to 1 for EVAL
        decoder_start_token_id=0,
        eos_token_id=model2.config.eos_token_id,
        pad_token_id=model2.config.eos_token_id,
        return_dict_in_generate=True,    # if commented the outputs of model.generate will be the sequence only
        output_scores=True,
        output_past_key_values=True,
        num_return_sequences=batch_size,
    )

    val = 0

    #f = open('.cache.txt', 'w')
    print('sampling ...')
    score1_all = []
    enc_text_all = []
    text_all = []

    for i in range(ceil(m / batch_size)):
        text, score1, enc_text , condprobs1= _prsample(prompt, model1, tokenizer1, gen_config1, seed)
        # for enc in enc_text:
        #     f.write(str(enc.item()))
        enc_text_all += enc_text
        score1_all += score1
        text_all += text
        seed += 1
    
    print('evaluating ...')
    score2_all = []
    for i in range(ceil(m / batch_size)):
        score1 = score1_all[i]
        enc_text = enc_text_all[i]
        currtext = text_all[i * batch_size : (i+1) * batch_size]
        score2, condprobs2 = _eval(prompt, model2, tokenizer2, currtext, gen_config2)
        score2_all += score2

    assert(len(score1_all) == len(score2_all))
    for i in range(len(score1_all)):
        score1 = score1_all[i]
        score2 = score2_all[i]
        if score1 > score2:
            val += (1 - (score2 / score1))
        dp.write("sample-" + str(i) + ":\t" + str(score1) + "\t" + str(score2) + "\n")
    dp.write("dTV: " + str(val/m) + "\n")
    dp.close()
    return val / m


def estimateKL(model1, model2, tokenizer1, tokenizer2, prompt, numsamples, dumpFile):
    '''
    estimates KL distance betwene model1 and model2 (temporary)
    '''

    # parameter setup
    seed = 10
    eps = 0.1
    delta = 0.2

    dp = open(dumpFile, "a")

    m = int(1 / eps**2 * _log(1 / delta)) + 1
    S = []

    # fix generation length
    ndim = 50 

    # gen config for _prsample
    gen_config1 = GenerationConfig(
        top_k=0,
        top_p=1,
        do_sample=True,
        #early_stopping=True,       #used for beam based method
        max_new_tokens = ndim,           # fix the generation length
        decoder_start_token_id=0,
        eos_token_id=model1.config.eos_token_id,
        pad_token_id=model1.config.eos_token_id,
        return_dict_in_generate=True,    # if commented the outputs of model.generate will be the sequence only
        output_scores=True,
        num_return_sequences=1,
    )


    # genconfig for _eval
    gen_config2 = GenerationConfig(
        top_k=0,
        top_p=1,
        do_sample=True,
        #early_stopping=True,       #used for beambased method
        max_new_tokens = 1,           # fix the generation length to 1 for EVAL
        decoder_start_token_id=0,
        eos_token_id=model2.config.eos_token_id,
        pad_token_id=model2.config.eos_token_id,
        return_dict_in_generate=True,    # if commented the outputs of model.generate will be the sequence only
        output_scores=True,
        num_return_sequences=1,
    )

    val = 0

    for i in track(range(m)):
        text, score1, enc_text = _prsample(prompt, model1, tokenizer1, gen_config1, seed)
        score2, prob_arr = _eval(prompt, model2, tokenizer2, enc_text, gen_config2, seed)
        if score1 > score2:
            val += (_log ((score1) / (score2)))
        seed += 1
        dp.write("sample-" + str(i) + ":\t" + str(score1) + "\t" + str(score2) + "\n")
    dp.close()

    return val / m


if __name__=='__main__':
    prompts = []
    with open('HumanEval.jsonl') as f:
        lines = f.readlines()
    for line in lines:
        j = json.loads(line)
        prompts.append(j['prompt'])
    prompt = prompts[0]
    modelid = 'stabilityai/stable-code-3b'#'gpt2'
    tokenizer1 = AutoTokenizer.from_pretrained(modelid)
    tokenizer2 = AutoTokenizer.from_pretrained(modelid)
    model1 = AutoModelForCausalLM.from_pretrained(modelid).to(torch_device) 
    model2 = model1#AutoModelForCausalLM.from_pretrained('stabilityai/stable-code-3b').to(torch_device)

    jobid = os.environ.get('PBS_JOBID', None)
    jobint = (jobid.split('.')[0] if jobid is not None else 0)
    if jobid is not None:
        jobint = 'j-' + jobint
        os.system('mkdir ' + jobint)
        filename = modelid.split("/")[-1]
        dumpFile = jobint + "/" + filename
        
    else:
        os.system('mkdir estimate-out')
        dumpFile = 'estimate.out'

    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(prompts) as prompts:

        print("GPU {}: {} prompts received".format(
            accelerator.process_index,
            len(prompts),
            ))
        dumpFile += str(accelerator.process_index)

        for prompt in prompts: 
            torch.cuda.empty_cache()
            print("dTV:", estimateTV(model1=model1, model2=model2, tokenizer1=tokenizer1, tokenizer2=tokenizer2, prompt=prompt, numsamples=None, dumpFile=dumpFile, batch_size=2))
            
