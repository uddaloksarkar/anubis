import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

src = '.'

from torch import cuda, cat
import argparse
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import gmpy2
import os, sys
import json
from accelerate import Accelerator
import logging
from tempfile import gettempdir
import glob
import numpy as np
import json

logger = logging.getLogger(__name__)
accelerator = Accelerator()

sys.path.insert(1, src)
from estimate import _eval, _evalwithchoices
from detokenize import get_collisions

torch_device = 'cuda' if cuda.is_available() else 'cpu'
gmpy2.set_context(gmpy2.context())


###########################################################
####                Reading Samples                     ###
###########################################################

def get_all_samples(dirname):
    allfilenames = [os.path.basename(fname) for fname in glob.glob(f"{dirname}/*")]
    allfilenames = sorted(allfilenames)
    samples = []
    sampleid = []
    for file in allfilenames:
        with open(f"{dirname}/{file}", 'r') as fp:
            samples.append(fp.read())
            sampleid.append(file)
    return samples, sampleid
        

##########################################################
# Parse input
##########################################################
parser = argparse.ArgumentParser()
parser.add_argument('--promptsrc', type=int,
                        default=0, dest='promptsrc', help="1 = TruthfulQA, 0 = HumanEval")
parser.add_argument('--num_prompts', type=int,
                        default=1, dest='num_prompts')
parser.add_argument('--taskID', type=int, 
                        default=0, dest='start_at')
parser.add_argument('--batch_size', type=int,
                        default=10, dest='batch_size')
parser.add_argument('--smpsrc', type=str, help="path to the sample files directory", dest='src')
parser.add_argument('--evalmodelID', type=int,
                        default=0, help="1 = deepseek, 0 = stability, 2 = gpt", dest='evalmodel')
parser.add_argument('--debug', type=int,
                        default=0, help="2 for memory debugging, 1 for normal Debugging, 0 else", dest='debug')
parser.add_argument('--verbose', type=int,
                        default=0, help="1 for verbose, 0 else", dest='verbose')
parser.add_argument('--nsamps', type=int,
                        default=None, dest='nsamps')
parser.add_argument('--topp', type=float,
                        default=1, dest='topp')
parser.add_argument('--temperature', type=float,
                        default=1, dest='temp')       
args = parser.parse_args()

num_prompts = args.num_prompts
start_at = args.start_at
batch_size = args.batch_size
samp_src = args.src
evalmodel = args.evalmodel
debug = args.debug
verb = args.verbose
nsamps = args.nsamps
promptsrc = args.promptsrc
topp = args.topp
temperature = args.temp

# memory debugging
if debug > 1:
    cuda.memory._record_memory_history(
            max_entries=100000
        )

##########################################
###         Data Loading            ######
##########################################
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

# load prompts
if promptsrc == 0:
    allprompts, taskid = loadData('HumanEval.jsonl')
elif promptsrc == 1:
    allprompts, taskid = loadData('domenicrosati/TruthfulQA')
else:
    raise NotImplementedError     

# setup model and tokenizer
if evalmodel == 0:
    evalmodelname = "stability"
    evalmodelid = 'stabilityai/stable-code-3b'
elif evalmodel == 1:
    evalmodelname = "deepseek"
    evalmodelid = 'deepseek-ai/deepseek-coder-1.3b-base'
elif evalmodel == 2:
    evalmodelname = "gpt"
    evalmodelid = 'gpt2'
elif evalmodel == 3:
    evalmodelname = "codellama"
    evalmodelid = 'codellama/CodeLlama-7b-Python-hf'
elif evalmodel == 4:
    evalmodelname = 'openllama'
    evalmodelid = 'openlm-research/open_llama_3b'
else:
    raise NotImplementedError
tokenizer = AutoTokenizer.from_pretrained(evalmodelid)
evalmodel = AutoModelForCausalLM.from_pretrained(evalmodelid).to(torch_device) 

# get all token collisions (needed for eval_with_choices)
collision_list = get_collisions(tokenizer)

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
    num_return_sequences= batch_size,
)

if verb:
    print('eval ...')

# setup output directory
corpusprefixFolder = 'data'
processrankStr = os.environ.get("OMPI_COMM_WORLD_RANK")
if (processrankStr == None):
    processrankStr = '0'
jobid = os.environ.get('PBS_JOBID', None)
jobint = (jobid.split('.')[0] if jobid is not None else 0)
jobFolder = f'{corpusprefixFolder}/j-{jobint}'
folder = samp_src 
evalfolder = jobFolder + '/' + evalmodelname + '/'
os.system('mkdir -p ' +jobFolder + ' ' + evalfolder)
os.system(f'echo Sample Source: {folder} > {evalfolder}/README.txt')

# which prompts to do 
promptbuff = allprompts[start_at:start_at+num_prompts]

# begin
accelerator.wait_for_everyone()
with accelerator.split_between_processes(promptbuff) as prompts:

    print("GPU {}: {} prompts received".format(
        accelerator.process_index,
        len(prompts),
        ))

    tmpFile = "{}/pid-{}".format(gettempdir(), accelerator.process_index)
    
    for prompt in prompts:
        task = taskid[allprompts.index(prompt)]
        cmd = 'find {}/{}/samples0{}.ds* > {}'.format(os.getcwd(), samp_src, task, tmpFile)
        # cmd = 'find ' + os.getcwd() + samp_src+ '/samples*'+ task + '* > ' + tmpFile
        os.system(cmd)
        print(f"Sample Source: {samp_src}")
        with open(tmpFile) as tmfp:
            lines = tmfp.readlines()
        if debug:
            print(tmpFile, lines)
        try:
            sampdumpFile = lines[0].strip()
        except IndexError:
            print("ERR", lines)
            continue
        if verb:
            print(sampdumpFile)
        encoded_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'][0].to(torch_device)
        scoredumpFile = f'{evalfolder}eval{processrankStr}{task}.ds_{evalmodelname}.json'
        occurdumpFile = f'{evalfolder}occur{processrankStr}{task}.ds_{evalmodelname}.json'

        # fscore = open(scoredumpFile, 'w'); fscore.close()
        if debug:
            encdumpDir = f'{evalfolder}enceval{processrankStr}{task}.ds_{evalmodelname}'
            condscoredumpDir = f'{evalfolder}condprobseval{processrankStr}{task}.ds_{evalmodelname}'
            os.system(f'mkdir -p {condscoredumpDir}')
            os.system(f'mkdir -p {encdumpDir}')

        
        samples, sampleid = get_all_samples(sampdumpFile)
        samples = samples[:nsamps]
        sampleid = sampleid[:nsamps]
        if verb:
            print(samples)
        evalscores = []
        sampleids = []
        if verb:
            print(sampleid)
        ccount = 0; ecount = 0
        for i in range(len(samples)//batch_size):
            samp_batch = samples[i*batch_size:min((i+1)*batch_size,len(samples))]
            sampleid_batch = sampleid[i*batch_size:min((i+1)*batch_size,len(samples))]
            if debug>1:
                try:
                    cuda.memory._dump_snapshot(f"checkmem-do-eval-probe2.pickle")
                except Exception as e:
                    logger.error(f"Failed to capture memory snapshot {e}")
            score, condprobs, enc= _evalwithchoices(prompt, evalmodel, tokenizer, samp_batch, gen_config, collision_list)
            evalscores += [float(iscore) for iscore in score]
            sampleids += samp_batch
            
            if verb:
                print(f"Encoding:{enc}, Probability:{score}")
            if debug:
                # store the encoding of the samples before detokenization
                for e in enc:
                    e = cat((encoded_prompt, e)).to('cpu').numpy().astype(int)
                    np.savetxt(f"{encdumpDir}/{ecount}", [e], delimiter = ',', fmt = '%d')
                    if verb:
                            print(e)
                    ecount += 1
                # store the condprobs of the samples when required    
                for s in condprobs:
                    with open(f"{condscoredumpDir}/{ccount}", "w") as fscore:
                        fscore.write(str(s)+ '\n')
                    ccount += 1
            
        if debug > 1:
            # Stop recording memory snapshot history.
            cuda.memory._record_memory_history(enabled=None)
        
        # with open(scoredumpFile, 'w') as fscore:
        #     for (score,ids) in zip(evalscores,sampleids):
        #         fscore.write(str(score)+ '\n')    
        empirical_dist = dict(zip(sampleids, evalscores))
        occurence = {}
        for samps in sampleids:
            occurence[samps] = occurence.get(samps, 0) + 1

        print(empirical_dist)
        print(evalscores)
        print(occurence)
        assert sum(occurence.values()) == len(sampleids)
        with open(scoredumpFile, 'w') as fscore:
            json.dump(empirical_dist, fscore)
        with open(occurdumpFile, 'w') as foccur:
            json.dump(occurence, foccur)
