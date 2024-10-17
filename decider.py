import os
from tempfile import gettempdir
from gmpy2 import mpfr
import argparse  
from  math import log10 as _log, floor
from matplotlib import pyplot as plt
from operator import itemgetter
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('--evalmodelID', type=int,
                        default=1, help="1 = deepseek, 0 = stability",dest='evalmodel')
parser.add_argument('--debug', type=int,
                        default=0, help="1 for Debugging, 0 else", dest='debug')
parser.add_argument('--verbose', type=int,
                        default=0, help="1 for verbosity, 0 else", dest='verbose')
parser.add_argument('--bwidth', type=int,
                        default=2, help="bucket width", dest='bwidth')
parser.add_argument('--lbsize', type=float,
                        default=0.05, help="fraction of samples to fall into lowest bucket", dest='thresh')
parser.add_argument('--sampthresh', type=int, default=500, help="minimum number of samples", dest='sampthresh')
parser.add_argument('--studir', type=str,dest='dir1')
parser.add_argument('--llmdir', type=str,dest='dir2')
parser.add_argument('--gthresh', type=float,dest='gthresh')
parser.add_argument('--lthresh', type=int,dest='lthresh')

args = parser.parse_args()
evalmodel = args.evalmodel
debug = args.debug
verb = args.verbose
bucketwdth = args.bwidth
thresh = args.thresh
gthresh = args.gthresh
lthresh = args.lthresh
dir1 = args.dir1
dir2 = args.dir2

class bucket:
    def __init__(self, logprob):
        self.samples = {}
        self.id = logprob
        self.count = 0

    def add_sample(self, sample, occur):
        self.samples[sample] = self.samples.get(sample, 0) + occur
        self.count += occur

    def get_prob(self):
        return self.count

class bucket_list:
    def __init__(self):
        self.buckets = {}
        self.nbucket = 0
        self.nsamples = 0

    def add_sample(self, sample, prob, occur):
        if prob <= 0: return
        id = floor(_log(prob)/_log(bucketwdth))
        if id < 1 - self.nbucket: id = 1 - self.nbucket
        self.buckets[id].add_sample(sample, occur)
        self.nsamples += occur

    def add_bucket(self, id):
        self.buckets[id] = bucket(id)
        self.nbucket += 1

    def normalize(self):
        for b in self.buckets:
            self.buckets[b].count = self.buckets[b].count / self.nsamples
            for s in self.buckets[b].samples:
                self.buckets[b].samples[s] = self.buckets[b].samples[s] / self.nsamples

    def get_lowestBucket(self):
        return self.buckets[1-self.nbucket].count
    
    def get_bucket(self, id):
        return self.buckets[id].count
    

def get_nbuckets(samplprobs, thresh):
    samplprobsorted = sorted(samplprobs)
    maxprob = samplprobsorted[int(thresh * len(samplprobsorted))]
    # get the number of buckets needed for the rest of elems 
    try:
        nbucket = - floor(_log(maxprob) / _log(bucketwdth)) + 1
    except ValueError:
        nbucket = 1
    print("c number of bucket:", nbucket)
    return nbucket 

def dkwtest(sampbuck, evalbuck):
    supreme = 0
    assert sampbuck.buckets.keys() == evalbuck.buckets.keys()
    sampcum = 0
    evalcum = 0
    for bn in sorted(sampbuck.buckets.keys())[1:]: # don't consider the lowest bucket
        sampcum += sampbuck.buckets[bn].count
        evalcum += evalbuck.buckets[bn].count
        diff = abs(sampcum - evalcum)
        supreme = max(supreme, diff)
    return supreme


def check_collisions(bucket1, bucket2):
    
    samples1 = bucket1.samples
    samples2 = bucket2.samples

    samples = set(samples1.keys()) | set(samples2.keys())
    for sample in samples:
        samples1.setdefault(sample, 0)
        samples2.setdefault(sample, 0)

    metric = 0
    for sample in samples:
        metric += (abs(samples1[sample] - samples2[sample])**2 / (samples1[sample] + samples2[sample]) -1)
    
    return metric


##############################
# Loading the data
##############################
print("c Loading the data ...")
if evalmodel == 0:
    evalmodelname = "stability"
    evalmodelid = 'stabilityai/stable-code-3b'
elif evalmodel == 1:
    evalmodelname = "deepseek"
    evalmodelid = 'deepseek-ai/deepseek-coder-1.3b-base'
else:
    raise NotImplementedError

# Hardcode for now
tmpfile = os.path.join(gettempdir(), "allevals")
parentdir = "."
srcdir1 = os.path.join(parentdir, dir1)
srcdir2 = os.path.join(parentdir, dir2)
dir1_name = os.path.basename(dir1)
dir2_name = os.path.basename(dir2)


cmd = f"find {os.path.join(srcdir1, 'eval*.ds_' + evalmodelname + '.json')} > {tmpfile}"
os.system(cmd)
with open(tmpfile) as tmp:
    allevalFiles = tmp.readlines()

##############################
# start bucketing
##############################
for evalfile in allevalFiles:
    
    # eval file
    filepos = evalfile.strip()
    
    with open(filepos) as fp:
        evaldata = json.load(fp)
    
    fileoccur = os.path.basename(filepos).replace("eval", "occur")
    fileoccur = os.path.join(srcdir1, fileoccur)
    with open(fileoccur) as fp:
        evaloccur = json.load(fp)

    evalprobs = evaldata.values()
    
    #sample file
    sampfilepos = filepos.replace(srcdir1, srcdir2)
    try:
        with open(sampfilepos) as fp:
            sampldata = json.load(fp)        
        samplprobs = sampldata.values()
    except FileNotFoundError:
        # if the corresponding file doesnot exist move to next
        continue

    fileoccur = os.path.basename(sampfilepos).replace("eval", "occur")
    fileoccur = os.path.join(srcdir2, fileoccur)
    with open(fileoccur) as fp:
        samploccur = json.load(fp)

    if len(evalprobs) == 0 or len(samplprobs) == 0 : continue

    earlyevalproblen = len(evalprobs)
    evalprobs = [prob for prob in evalprobs if prob != 0]  # Remove zeros from evalprobs
    earlysamplproblen = len(samplprobs)
    samplprobs = [prob for prob in samplprobs if prob != 0]  # Remove zeros from samplprobs

    # if low samples do skip
    if min(len(evalprobs), len(samplprobs)) < args.sampthresh: 
        print(filepos, f"c evalprob lens falls down form {earlyevalproblen} to {len(evalprobs)}, and sampleprob len from {earlysamplproblen} to {len(samplprobs)}") 
        continue 
    print('c filename:',os.path.basename(filepos))
    
    # hardcoding the prompt id
    promptID = sampfilepos.split("/")[-1].split(".")[0]
    
    # only take those samples whose evals are done
    nsamp = min(len(evalprobs), len(samplprobs))
    random.shuffle(evalprobs)
    random.shuffle(samplprobs)
    evalprobs = evalprobs[:nsamp]
    samplprobs = samplprobs[:nsamp]

    # find the probability of the lowest bucket and number of buckets to bucket the rest
    try:
        nbucket = get_nbuckets(samplprobs, thresh)
    except IndexError:
        continue

    # bucketing
    sampbins = bucket_list()
    evalbins = bucket_list()
    for bcks in range(nbucket):
        sampbins.add_bucket(-1 * bcks)
        evalbins.add_bucket(-1 * bcks)
    for i, prob in sampldata.items():
        sampbins.add_sample(i, prob, samploccur[i])
    for i, prob in evaldata.items():
        evalbins.add_sample(i, prob, evaloccur[i])
    
    # check for collisions
    max_coll = 0; b_id = None
    for (bucket1, bucket2) in zip(sampbins.buckets.values(), evalbins.buckets.values()):
        assert bucket1.id == bucket2.id
        if max_coll < check_collisions(bucket1, bucket2):
            b_id = bucket2.id
        max_coll = max(max_coll, check_collisions(bucket1, bucket2))
        # print(f'bucket ids [{bucket2.id}] ->', check_collisions(bucket1, bucket2))
    print(f'c max bucket ids [{b_id}] ->', max_coll)

    sampbins.normalize()
    evalbins.normalize()

    #DKW test on the bucket distribution
    dkweval = dkwtest(sampbins, evalbins)
    print(f"c dkw: {os.path.basename(evalfile).strip()} ---> {dkweval}")

    if dkweval < gthresh  and max_coll < lthresh: 
        print(f"\033[91m FLAGGED \033[0m")
    else:
        print(f"\033[92m PASSED \033[0m")