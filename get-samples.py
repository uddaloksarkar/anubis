import os
import torch
import argparse
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import gmpy2
import os, sys
import json
from math import ceil
from accelerate import Accelerator
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
accelerator = Accelerator()

src = '.'
sys.path.insert(1, src)
from estimate import _prsample

#locale.getpreferredencoding = lambda: "UTF-8"
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
gmpy2.set_context(gmpy2.context())


# Parse CL inputs
parser = argparse.ArgumentParser()
parser.add_argument('--promptsrc', type=int,
                        default=0, dest='promptsrc', help="1 = TruthfulQA, 0 = HumanEval")
parser.add_argument('--num_prompts', type=int,
                        default=1, dest='num_prompts')
parser.add_argument('--start_at', type=int, 
                        default=0, dest='start_at')
parser.add_argument('--samples', type=int,
                        default=30, dest='m')
parser.add_argument('--b', type=int,
                        default=10, dest='batch_size')
parser.add_argument('--model', type=int,
                        default=0, help="2 = gpt, 1 = deepseek, 0 = stability",dest='model')
parser.add_argument('--debug', type=int,
                        default=0, help="1 for Normal Debugging, 2 for CudaMemory Debugging, 0 for off", dest='debug')
parser.add_argument('--verbose', type=int,
                        default=0, help="1 for verbosity, 0 else", dest='verbose')
parser.add_argument('--ndim', type=int,
                        default=100, dest='ndim')
parser.add_argument('--seed', type=int,
                        default=10, dest='seed')
parser.add_argument('--topp', type=float,
                        default=1, dest='topp')       
parser.add_argument('--temperature', type=float,
                        default=1, dest='temp')                        
args = parser.parse_args()
num_prompts = args.num_prompts
start_at = args.start_at
m = args.m
batch_size = args.batch_size
model = args.model
debug = args.debug
verb = args.verbose
ndim = args.ndim 
seed = args.seed
topp = args.topp
temperature = args.temp
promptsrc = args.promptsrc

##########################################
###         Data Loading            ######
##########################################
## uncomment to load the HumanEval.json :
print("data loading...")
def loadData(promptsource):
    allprompts = []
    taskid = []
    if promptsource == 'HumanEval.jsonl':
        with open(promptsource) as f:
            lines = f.readlines()
        for line in lines:
            j = json.loads(line)
            allprompts.append(j['prompt'])
            taskid.append(j['task_id'].split("/")[-1])
        return allprompts,taskid
    else:
        raise NotImplementedError

if promptsrc == 0:
    allprompts, taskid = loadData('HumanEval.jsonl')
    sourcename = 'HumanEval'
elif promptsrc == 1:
    allprompts, taskid = loadData('domenicrosati/TruthfulQA')
    sourcename = 'TruthfulQA'
print(f"data loaded from {sourcename}")
     
def getDirName(outdir, processrankStr, task, modelname, dirType):
    retStr = f'{outdir}/{dirType}{processrankStr}{task}.ds_{modelname}'
    if dirType == 'score':
        f = open(retStr, "w"); f.close()
    else:
        os.system(f'mkdir -p {retStr}')
    return retStr

# debug for memory usage
if debug == 2:
    print("Memory debugging enabled")
    torch.cuda.memory._record_memory_history(
            max_entries=100000
    )

# model and tokenizer loading 
modelid, modelname = "",""
if model == 0:
    modelname = "stability"
    modelid = 'stabilityai/stable-code-3b'
elif model == 1:
    modelname = "deepseek"
    modelid = 'deepseek-ai/deepseek-coder-1.3b-base'
elif model == 2:
    modelname = "codegemma"
    modelid = 'google/codegemma-2b'
else:
    raise NotImplementedError
print(f"loading model {modelname} ...")
tokenizer = AutoTokenizer.from_pretrained(modelid)
model = AutoModelForCausalLM.from_pretrained(modelid).to(torch_device) 
print(f"model {modelname} loaded from {modelid}")

# setup for samples directory
corpusprefixFolder = 'data'
processrankStr = os.environ.get("OMPI_COMM_WORLD_RANK")
if processrankStr is None:
    processrankStr = '0'
jobid = os.environ.get('PBS_JOBID', None)
jobint = (jobid.split('.')[0] if jobid is not None else 0)
jobFolder = f'{corpusprefixFolder}/j-{jobint}'
os.system(f'mkdir -p {jobFolder}')
outdir = f'{jobFolder}/{modelname}'
os.system(f'mkdir -p {outdir}')

# Stop Tokens 
class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_word in self.stop_words:
            if stop_word in decoded_output:
                return True
        return False


# class CustomStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, max_length=50, stop_strings=None, prompt=None):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.stop_strings = stop_strings if stop_strings is not None else []
#         self.stopped = [False] * max_length  # Track stopped sequences
#         self.prompt = prompt

#     def __call__(self, input_ids, scores, **kwargs):
#         batch_size = input_ids.shape[0]
#         for i in range(batch_size):
#             if self.stopped[i]:
#                 continue  # Skip already stopped sequences
#             if len(input_ids[i]) >= self.max_length:
#                 self.stopped[i] = True
#             else:
#                 decoded_seq = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
#                 if any(stop_string in decoded_seq for stop_string in self.stop_strings):
#                     self.stopped[i] = True
#         return all(self.stopped) #any

# stop_words = ["\n\n#", "\n\ndef"]
# stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'] for stop_word in stop_words]
# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
# custom_stopping_criteria = CustomStoppingCriteria(
#     max_length=500,
#     stop_strings=["\n\n#", "\n\ndef"]
# )
# stopping_criteria=StoppingCriteriaList([custom_stopping_criteria])

# gen config for _prsample
gen_config = GenerationConfig(
    temperature=temperature,
    top_k=0,
    top_p=topp,
    do_sample=True,
    max_new_tokens = ndim,           # fix the generation length
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.eos_token_id,
    return_dict_in_generate=True,    # if commented the outputs of model.generate will be the sequence only
    output_scores=True,
    num_return_sequences= batch_size,
)

print('start sampling ...')
promptbuff = allprompts[start_at:start_at+num_prompts]

accelerator.wait_for_everyone()
with accelerator.split_between_processes(promptbuff) as prompts:

    print(f"GPU {accelerator.process_index}: {len(prompts)} prompts received")

    for prompt in prompts:
        task = taskid[allprompts.index(prompt)]
        sampdumpDir = getDirName(outdir, processrankStr, task, modelname, 'samples')
        scoredumpFile = getDirName(outdir, processrankStr, task, modelname, 'score')
        if debug:
            encdumpDir = getDirName(outdir, processrankStr, task, modelname, 'enc')
            condscoredumpDir = getDirName(outdir, processrankStr, task, modelname, 'condprobs')
        
        stop_words=["\n\ndef"]
        custom_stopping_criteria = StopWordsCriteria(stop_words, tokenizer)
        # CustomStoppingCriteria(
        # tokenizer=tokenizer, 
        # max_length=500,
        # stop_strings=["\n\n#", "\n\ndef"],
        # prompt=prompt
        # )
        stopping_criteria=None #StoppingCriteriaList([custom_stopping_criteria])
        
        # keep metadata of sample generation
        with open(outdir+'/metadata.txt', "w") as fmeta:
            fmeta.write(f'top_p: {topp}\n top_k: 0\n temperature: {temperature}\n generation dimension: {ndim}\n')

        # todo: random naming scheme for files
        tcount = 0; ecount = 0; ccount = 0
        for _ in range(ceil(m / batch_size)):
            seed += 1

            if debug == 2:
                try:
                    torch.cuda.memory._dump_snapshot(f"checkmem-get-samples.pickle")
                except Exception as e:
                    print(f"Failed to capture memory snapshot {e}")

            text, score, enc_text, condprobs = _prsample(prompt, model, tokenizer, gen_config, stopping_criteria, seed)

            # store the samples
            for t in text:
                with open(f"{sampdumpDir}/{tcount}.py", "w") as fsamp:
                    fsamp.write(t)
                tcount += 1
            # store the probs of the samples
            with open(scoredumpFile, "a") as fscore:
                for s in score:
                    if verb:
                        print(s)
                    fscore.write(str(s)+ '\n')
            if debug:
                # store the encoding of the samples before detokenization
                for e in enc_text:
                    e = e.to('cpu').numpy().astype(int)
                    np.savetxt(f"{encdumpDir}/{ecount}", [e], delimiter = ',', fmt = '%d')
                    if verb:
                            print(e)
                    ecount += 1
                # store the condprobs of the samples when required    
                for s in condprobs:
                    with open(f"{condscoredumpDir}/{ccount}", "w") as fscore:
                        fscore.write(str(s)+ '\n')
                    ccount += 1
            
# make debugging more easy
if m == 1 and debug:
    print(text); print(score)
    with open(f'fdebugsamples.txt', 'w') as f:
        f.write(text[0])
    with open('fdebugenc.txt', 'w') as f:
        f.write(str(enc_text[0]))
    with open('fdebugcond.txt', 'w') as f:
        f.write(str(condprobs[0]))

if debug == 2:        
    torch.cuda.memory._record_memory_history(enabled=None)
