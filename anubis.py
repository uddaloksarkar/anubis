import os
import torch
import argparse
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import gmpy2
import os, sys
import json
from math import ceil
import numpy as np
import glob
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

src = '.'
sys.path.insert(1, src)
from estimate import _prsample, _eval, _evalwithchoices
from detokenize import get_collisions
from semantic_check import truncate_on_stop_and_return, check_correctness
from get_bin_plots import get_nbuckets, check_collisions, dkwtest, bucket_list

#locale.getpreferredencoding = lambda: "UTF-8"
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
gmpy2.set_context(gmpy2.context())

###########################################################
####                Reading Samples                     ###
###########################################################

def get_all_samples(dirname):
    allfilenames = [os.path.basename(fname) for fname in glob.glob(f"{dirname}/*")]
    allfilenames = sorted(allfilenames)
    samples = []
    studentid = []
    for file in allfilenames:
        with open(f"{dirname}/{file}", 'r') as fp:
            samples.append(fp.read())
            studentid.append(file)
    return samples, studentid

def get_all_json_samples(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    studentid = list(data.keys())
    samples = list(data.values())
    return samples, studentid

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
    
def getDirName(outdir, processrankStr, task, modelname, dirType):
    retStr = f'{outdir}/{dirType}{processrankStr}{task}.ds_{modelname}'
    if dirType == 'score':
        f = open(retStr, "w"); f.close()
    else:
        os.system(f'mkdir -p {retStr}')
    return retStr

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


# Parse CL inputs
parser = argparse.ArgumentParser()
parser.add_argument('--smpsrc', type=str, help="path to the cohort directory", dest='src')
parser.add_argument('--promptsrc', type=str,
                        dest='promptsrc', help="prompt source file")
parser.add_argument('--promptid', type=int,
                        default=1, dest='promptid', help="prompt id in the source file")
parser.add_argument('--nsamps', type=int,
                        default=30, dest='nsamps')
parser.add_argument('--sampb', type=int,
                        default=10, dest='samp_batch_size')
parser.add_argument('--evalb', type=int,
                        default=1, dest='eval_batch_size')
parser.add_argument('--model', type=int,
                        default=0, help="2 = codegemma, 1 = deepseek, 0 = stability",dest='model')
parser.add_argument('--verbose', type=int,
                        default=0, help="1 for verbosity, 0 else", dest='verbose')
parser.add_argument('--ndim', type=int,
                        default=100, dest='ndim')
parser.add_argument('--seed', type=int,
                        default=10, dest='seed')
parser.add_argument('--topp', type=float,
                        default=0.95, dest='topp')       
parser.add_argument('--temperature', type=float,
                        default=0.8, dest='temp') 
parser.add_argument('--thresh', type=float,
                        default=0.08, help="lower bucket threshold in percentile", dest='thresh')
parser.add_argument('--sampthresh', type=int, default=500,          
                        help="minimum number of samples", dest='sampthresh')
parser.add_argument('--bwidth', type=int,
                        default=2, help="bucket width", dest='bwidth')
args = parser.parse_args()
cohort_src = args.src
promptid = args.promptid
nsamps = args.nsamps
samp_batch_size = args.samp_batch_size
eval_batch_size = args.eval_batch_size
model = args.model
verb = args.verbose
ndim = args.ndim 
seed = args.seed
topp = args.topp
temperature = args.temp
promptsrc = args.promptsrc
thresh = args.thresh
bucketwdth = args.bwidth

##########################################
###         Data Loading            ######
##########################################
## uncomment to load the HumanEval.json :
print("data loading...")

allprompts, taskid = loadData('HumanEval.jsonl')
sourcename = 'HumanEval'

prompt = allprompts[promptid]

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
    num_return_sequences= samp_batch_size,
)

print('start sampling ...')

task = taskid[allprompts.index(prompt)]
stop_words=["\n\ndef"]
custom_stopping_criteria = StopWordsCriteria(stop_words, tokenizer)
stopping_criteria=None #StoppingCriteriaList([custom_stopping_criteria])

# keep metadata of sample generation
with open('metadata.txt', "w") as fmeta:
    fmeta.write(f'top_p: {topp}\n top_k: 0\n temperature: {temperature}\n generation dimension: {ndim}\n')

# sampling
tcount = 0; ecount = 0; ccount = 0
sample_text, sample_score = [], []
for _ in tqdm(range(ceil(nsamps / samp_batch_size)), desc="Sampling"):
    seed += 1
    text, _, _, _ = _prsample(prompt, model, tokenizer, gen_config, stopping_criteria, seed)
    sample_text += text

# sanitization
print('sanitizing samples ...')
sanitized_text = []
for i, text_sample in enumerate(sample_text):
    truncated_text = truncate_on_stop_and_return(text_sample, prompt)
    # future = check_correctness(prompt, truncated_text, timeout=3.0, completion_id=i)
    # if future['passed']: ccount += 1
    sanitized_text.append(truncated_text)
# print(f"Sample success rate: {ccount} out of {len(sample_text)}")

# setup evalmodel and tokenizer
evaltokenizer = tokenizer
evalmodel = model

# get all token collisions (needed for eval_with_choices)
collision_list = get_collisions(evaltokenizer)

# gen config for _eval
gen_config = GenerationConfig(
    top_k=0,
    top_p=topp,
    temperature=temperature,
    do_sample=True,
    max_new_tokens = 1,           # fix the generation length to 1 for eval
    decoder_start_token_id=0,
    eos_token_id=evalmodel.config.eos_token_id,
    pad_token_id=evalmodel.config.eos_token_id,
    return_dict_in_generate=True,    # if commented the outputs of model.generate will be the sequence only
    output_scores=True,
    num_return_sequences= eval_batch_size,
)

# eval students
print(f"cohort Source: {cohort_src}")
# students, studentid = get_all_samples(cohort_src)
students, studentid = get_all_json_samples(cohort_src)
students = students[:nsamps]
evalscores = []
studentids = []
ccount = 0; ecount = 0
for i in tqdm(range(len(students)//eval_batch_size), desc="Evaluating students"):
    samp_batch = students[i * eval_batch_size : min((i+1)*eval_batch_size, len(students))]
    score, _, _ = _eval(prompt, evalmodel, evaltokenizer, samp_batch, gen_config, collision_list)
    evalscores += [float(iscore) for iscore in score]
    studentids += samp_batch
    
empirical_student = dict(zip(studentids, evalscores))
occurence_student = {}
for samps in studentids:
    occurence_student[samps] = occurence_student.get(samps, 0) + 1

if verb:
    for key, value in empirical_student.items():
        print('>'*60 , '\n'*3, key, '\n'*3, '>'*60)
    print(empirical_student.values())
    print(occurence_student.values())

assert sum(occurence_student.values()) == len(studentids)

# eval samples
sanitized_text = sanitized_text[:nsamps]
evalscores = []
sampleids = []
ccount = 0; ecount = 0
for i in tqdm(range(len(sanitized_text)//eval_batch_size), desc="Evaluating samples"):
    samp_batch = sanitized_text[i*eval_batch_size:min((i+1)*eval_batch_size,len(sanitized_text))]
    score, _, _ = _eval(prompt, evalmodel, evaltokenizer, samp_batch, gen_config, collision_list)
    evalscores += [float(iscore) for iscore in score]
    sampleids += samp_batch
    
empirical_sample = dict(zip(sampleids, evalscores))
occurence_sample = {}
for samps in sampleids:
    occurence_sample[samps] = occurence_sample.get(samps, 0) + 1

if verb:
    for key, value in empirical_sample.items():
        print('>'*60 , '\n'*3, key, '\n'*3, '>'*60)
    print(empirical_sample.values())
    print(occurence_sample.values())

assert sum(occurence_sample.values()) == len(sampleids)

#start evaluation
print("start evaluation ...")
empirical_student_probs = empirical_student.values()
empirical_sample_probs = empirical_sample.values()    
if len(empirical_sample_probs) == 0 or len(empirical_student_probs) == 0 : raise ValueError("Empty list of probabilities")

earlyevalproblen = len(empirical_sample_probs)
empirical_sample_probs = [prob for prob in empirical_sample_probs if prob != 0]  # Remove zeros from empirical_sample_probs
earlysamplproblen = len(empirical_student_probs)
empirical_student_probs = [prob for prob in empirical_student_probs if prob != 0]  # Remove zeros from empirical_student_probs

# if low samples do skip
if min(len(empirical_sample_probs), len(empirical_student_probs)) < args.sampthresh: 
    print(f"evalprob lens falls down form {earlyevalproblen} to {len(empirical_sample_probs)}, and sampleprob len from {earlysamplproblen} to {len(empirical_student_probs)}") 

# find the probability of the lowest bucket and number of buckets to bucket the rest
try:
    nbucket = get_nbuckets(empirical_student_probs, thresh, bucketwdth)
except IndexError:
    print("IndexError: Empty list of probabilities")

# bucketing
sampbins = bucket_list()
evalbins = bucket_list()
for bcks in range(nbucket):
    sampbins.add_bucket(-1 * bcks)
    evalbins.add_bucket(-1 * bcks)
for i, prob in empirical_student.items():
    sampbins.add_sample(i, prob, occurence_student[i], bucketwdth)
for i, prob in empirical_sample.items():
    evalbins.add_sample(i, prob, occurence_sample[i], bucketwdth)

# check for collisions
max_coll = 0; b_id = None
for (bucket1, bucket2) in zip(sampbins.buckets.values(), evalbins.buckets.values()):
    assert bucket1.id == bucket2.id
    if max_coll < check_collisions(bucket1, bucket2):
        b_id = bucket2.id
    max_coll = max(max_coll, check_collisions(bucket1, bucket2))
print(f'max bucket ids [{b_id}] ->', max_coll)

sampbins.normalize()
evalbins.normalize()

# low bucket test
lowbuck = abs(sampbins.get_lowestBucket() - evalbins.get_lowestBucket())
print(f"lb: ---> {lowbuck}")

#DKW test on the bucket distribution
dkweval = dkwtest(sampbins, evalbins)
print(f"dkw: ---> {dkweval}")

if max_coll < 30 and dkweval < 0.08:
    print("ACCEPT")
else:
    print("REJECT")