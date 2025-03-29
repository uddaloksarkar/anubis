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
            collision[tuple(retok)] = [tok]#, gen_text]
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
    '''
    enc = tiktoken.get_encoding("cl100k_base")
    nclash = 0
    edges = []

    if tok is None:
        for tok in track(range(50152)):
            gen_text = tokenizer.batch_decode([tok],  skip_special_tokens = True)
            # gen_text = enc.decode([tok])
            try:
                detok = tokenizer(gen_text)['input_ids'][0][-1]
                # detok = enc.encode(gen_text)
            except IndexError:
                print("end tok:", tok, '\n', gen_text)
                continue
            #     break
            if gen_text == ['']: break
            for txt in gen_text: 
                txt1 = [ord(char) for char in txt]
            if tok != detok:
                edges.append((tok,detok))
                print(tok, "==##==", detok, "==##==", txt)#, "==##==", txt1)
                nclash += 1
        print(nclash)
        print("fracclashes:", nclash/tok)
        # gen_text = tokenizer.batch_decode([tok])
        # print(gen_text)
        # print(tokenizer(gen_text)['input_ids'])

        # Create a directed graph
        G = nx.DiGraph()

        # Add edges
        G.add_edges_from(edges)

        # Visualize the graph
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, node_size=1, node_color="skyblue", arrows=True)

        # Show the graph
        plt.savefig('tokdag.png')

        if has_cycle := nx.recursive_simple_cycles(G):
            print("The graph contains cycles:", has_cycle)
        else:
            print("The graph does not contain any cycles.")

        # Find the longest path
        longest_path = nx.algorithms.dag.dag_longest_path(G)
        print("The longest path in the graph is:", longest_path)

        print("number of connected components:", nx.number_strongly_connected_components(G))
    else:

        # gen_text = enc.decode([tok])
        # detok = enc.encode(gen_text)
        gen_text = tokenizer.batch_decode([tok],  skip_special_tokens = True)
        try:
            detok = tokenizer(gen_text)['input_ids'][0][-1]
            print("tok:", tok, '\n', gen_text, '\n detok:', detok)
        except IndexError:
            print("end tok:", tok, '\n', gen_text)
        '''