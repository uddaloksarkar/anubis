from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import contextlib
import itertools
from rich.progress import track
import locale
from tqdm.contrib.itertools import product
import networkx as nx
import matplotlib.pyplot as plt        

locale.getpreferredencoding = lambda: "UTF-8"


def get_collisions(tokenizer):
    # counts all the collisions
    maxtok = len(tokenizer)
    collision = {}
    for tok in range(maxtok):
        gen_text = tokenizer.batch_decode([[tok]], skip_special_token = True)
        retok = tokenizer(gen_text, add_special_tokens = False)['input_ids'][0]
        try:
            tokenizer_name = tokenizer.name_or_path.split('/')[1]
            if tokenizer_name == 'open_llama_3b' and retok[0] == 31822:
                continue
        except IndexError:
            print(retok)
        if [tok] != retok:
            collision[tuple(retok)] = [tok, gen_text]
    with contextlib.suppress(KeyError):
        collision.pop(())
    return collision

def get_2collisions(tokenizer):
    # counts all the collisions
    maxtok = len(tokenizer)
    collision = {}
    for tok1, tok2 in product(range(maxtok), range(maxtok)):
        gen_text = tokenizer.batch_decode([[tok1, tok2]])
        retok = tokenizer(gen_text)['input_ids'][0]
        if [tok1, tok2] != retok:
            if (tok1, tok2) in collision:
                collision[(tok1, tok2)].append(retok)
            else:
                collision[(tok1, tok2)] = [retok]
            print(collision)
    return collision


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int,
                            default=0, help="1 = deepseek, 0 = stability, 2 = gpt", dest='model')
    parser.add_argument('--tok', type=int,
                            default=None, dest='tok')


    args = parser.parse_args()

    model = args.model
    tok = args.tok

    if model == 0:
        modelname = "stability"
        modelid = 'stabilityai/stable-code-3b'
    elif model == 1:
        modelname = "deepseek"
        modelid = 'deepseek-ai/deepseek-coder-1.3b-base'
    elif model == 2:
        modelname = "gpt"
        modelid = 'gpt2'
    elif model == 4:
        modelname = 'openllama'
        modelid = 'openlm-research/open_llama_3b'
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(modelid)
    collision = get_collisions(tokenizer)
    print(collision, len(collision))
    maxlen = 0 #max([len(l) for l in collision[key]])
    for key in collision:
        if len(key) > maxlen: 
            maxlen = len(key)
        if len(key) > 1:
            print(key, collision[key])
    print(maxlen)
    print(len(collision))