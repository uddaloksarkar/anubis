import os
from tempfile import gettempdir
from gmpy2 import mpfr
import argparse  
from  math import log10 as _log, floor, ceil
from matplotlib import pyplot as plt
from operator import itemgetter
import random
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc


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
    if verb: print("number of bucket:", nbucket)
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

def equalize(A, B, A1, B1):
    len_B = len(B)
    len_A = len(A)
    if len_A > len_B:
        larger_dict, smaller_dict = A, B
        larger_dict1, smaller_dict1 = A1, B1
    else:
        larger_dict, smaller_dict = B, A
        larger_dict1, smaller_dict1 = B1, A1
        
    assert larger_dict.keys() == larger_dict1.keys()

    # Calculate how many elements to remove from the larger dictionary
    num_to_remove = len(larger_dict) - len(smaller_dict)

    # Randomly select keys to remove from the larger dictionary
    keys_to_remove = random.sample(list(larger_dict.keys()), num_to_remove)

    # Remove the selected key-value pairs from the larger dictionary
    for key in keys_to_remove:
        del larger_dict[key]
        del larger_dict1[key]

    return A, B, A1, B1    


def lazy_mixing(A, B, A1, B1, percentage):
    '''Used for lazy mixing of at evaluation time without creating new datasets '''

    len_B = len(B)
    
    num_replacements = ceil((percentage / 100) * len_B)

    # Get a list of keys in B and A
    keys_B = list(B.keys())
    keys_A = list(A.keys())

    # Randomly choose keys from B to replace
    keys_to_replace_in_B = random.sample(list(range(len_B)), num_replacements)
    keys_to_keep_in_B = list(set(list(range(len_B))) - set(keys_to_replace_in_B))

    newB, newB1 = {}, {}
    # Replace the randomly chosen keys in B
    for i in keys_to_replace_in_B:
        if i < len(keys_A):  # Replace with values from A, using A's keys
            key_A = keys_A[i]
            newB[key_A] = A[key_A]  # Replace value in B with corresponding value from A
            newB1[key_A] = A1[key_A]  # Replace value in B with corresponding value from A
        else:
            # assert False
            continue  # Stop if we've run out of keys in A to replace with
    for i in keys_to_keep_in_B:
        if i < len(keys_B):  
            key_B = keys_B[i]
            newB[key_B] = B[key_B]  
            newB1[key_B] = B1[key_B]  
        else:
            assert False

    return newB, newB1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bwidth', type=int, default=2,
                        help="bucket width", dest='bwidth')
    parser.add_argument('--debug', type=int, default=0,
                        help="1 for Debugging, 0 else", dest='debug')
    parser.add_argument('--eps1', type=int, dest='perc1',
                        help="student plagiarism lower percentage")
    parser.add_argument('--eps2', type=int, dest='perc2',
                        help="student plagiarism upper percentage")
    parser.add_argument('--evalmodel', type=int, default=1,
                        help="2 = codegemma, 1 = deepseek, 0 = stability", dest='evalmodel')
    parser.add_argument('--origllm', type=str, dest='origllm',
                        help="llm samples")
    parser.add_argument('--origstu', type=str, dest='origstu',
                        help="orignal student samples")
    parser.add_argument('--sampthresh', type=int, default=500,
                        help="minimum number of samples", dest='sampthresh')
    parser.add_argument('--samps', type=int, dest='nsamps',         default=3000,
                        help="n-samples")
    parser.add_argument('--seed', type=int, dest='seed', default=420,
                        help="seed")
    parser.add_argument('--stucorrupt', type=str, dest='stucorrupt',
                        help="llm in disguise of student samples")
    parser.add_argument('--thresh', type=float, default=0.05,
                        help="lower bucket threshold in percentile", dest='thresh')
    parser.add_argument('--threshin', type=int, dest='thresh2', default=100,
                        help="inbucket threshold")
    parser.add_argument('--threshout', type=float, dest='thresh1', default=0.08,
                        help="dkw threshold")
    parser.add_argument('--verbose', type=int, default=1,
                        help="1 for verbosity, 0 else", dest='verbose')


    args = parser.parse_args()
    evalmodel = args.evalmodel
    debug = args.debug
    verb = args.verbose
    bucketwdth = args.bwidth
    thresh = args.thresh
    origstu = args.origstu
    stucorrupt = args.stucorrupt
    origllm = args.origllm
    # llmcorrupt = args.llmcorrupt
    percentage1 = args.perc1
    percentage2 = args.perc2
    thresh1 = args.thresh1
    thresh2 = args.thresh2
    nsamps = args.nsamps
    assert int(percentage1) <= int(percentage2)
    random.seed(args.seed)

    ##############################
    # Loading the data
    ##############################
    if verb: print("Loading the data ...")
    if evalmodel == 0:
        evalmodelname = "stability"
        evalmodelid = 'stabilityai/stable-code-3b'
    elif evalmodel == 1:
        evalmodelname = "deepseek"
        evalmodelid = 'deepseek-ai/deepseek-coder-1.3b-base'
    elif evalmodel == 2:
        evalmodelname = "codegemma"
        evalmodelid = 'google/codegemma-2b'
    else:
        raise NotImplementedError

    # Hardcode for now
    tmpfile = os.path.join(gettempdir(), "allevals")
    parentdir = "."
    srcdir1 = os.path.join(parentdir, origstu)
    srcdir2 = os.path.join(parentdir, origllm)
    srcdir3 = os.path.join(parentdir, stucorrupt)
    dir1_name = os.path.basename(origstu)
    dir2_name = os.path.basename(origllm)
    dir3_name = os.path.basename(stucorrupt)


    cmd = f"find {os.path.join(srcdir1, 'eval*.ds_' + evalmodelname + '.json')} > {tmpfile}"
    os.system(cmd)
    with open(tmpfile) as tmp:
        allevalFiles = tmp.readlines()

    tpos, tneg, fpos, fneg = 0, 0, 0, 0
    ##############################
    # start bucketing
    ##############################

    allflagres = []
    allpassres = []

    for evalfile in allevalFiles:
        
        # student eval,occur files
        filepos = evalfile.strip()
        with open(filepos) as fp:
            evaldata = json.load(fp)
        evaldata = dict(random.sample(sorted(evaldata.items()), min(nsamps, len(evaldata))))

        fileoccur = os.path.basename(filepos).replace("eval", "occur")
        fileoccur = os.path.join(srcdir1, fileoccur)
        with open(fileoccur) as fp:
            evaloccur = json.load(fp)
        evaloccur = {key: evaloccur[key] for key in evaldata.keys() if key in evaloccur}
            

        # llm files to corrupt student samples set
        stucorruptpos = filepos.replace(srcdir1, srcdir3)
        try:
            with open(stucorruptpos) as fp:
                stucorruptdata = json.load(fp)        
        except FileNotFoundError:
            # if the corresponding file doesnot exist move to next
            continue
        fileoccur = os.path.basename(stucorruptpos).replace("eval", "occur")
        fileoccur = os.path.join(srcdir3, fileoccur)
        with open(fileoccur) as fp:
            stucorruptoccur = json.load(fp)

        # llm eval,occur files
        sampfilepos = filepos.replace(srcdir1, srcdir2)
        try:
            with open(sampfilepos) as fp:
                sampldata = json.load(fp)
            sampldata = dict(random.sample(sorted(sampldata.items()), min(nsamps, len(sampldata))))        
        except FileNotFoundError:
            # if the corresponding file doesnot exist move to next
            continue
        fileoccur = os.path.basename(sampfilepos).replace("eval", "occur")
        fileoccur = os.path.join(srcdir2, fileoccur)
        with open(fileoccur) as fp:
            samploccur = json.load(fp)
        samploccur = {key: samploccur[key] for key in sampldata.keys() if key in samploccur}

        # hardcoding the prompt id
        promptID = sampfilepos.split("/")[-1].split(".")[0]

        # lazymixing: student files are corrupted (mixed) with llm files
        evaldata1, evaloccur1 = lazy_mixing(stucorruptdata, evaldata, stucorruptoccur, evaloccur, percentage1)
        evaldata2, evaloccur2 = lazy_mixing(stucorruptdata, evaldata, stucorruptoccur, evaloccur, percentage2)

        # identifies llm generated text
        evalprobs = evaldata.values()
        earlyevalproblen = len(evalprobs)
        evalprobs = [prob for prob in evalprobs if prob != 0] 
        samplprobs = sampldata.values()
        if len(evalprobs) == 0 or len(samplprobs) == 0 : continue
        earlysamplproblen = len(samplprobs)
        samplprobs = [prob for prob in samplprobs if prob != 0]  # Remove zeros from samplprobs
        # if low samples do skip
        if min(len(evalprobs), len(samplprobs)) < args.sampthresh: 
            if verb: print(filepos, f"evalprob lens falls down form {earlyevalproblen} to {len(evalprobs)}, and sampleprob len from {earlysamplproblen} to {len(samplprobs)}") 
            continue 
        if verb: print('filename:',os.path.basename(filepos))
        
        # identifies student generated text
        evalprobs1 = evaldata1.values()    
        earlyevalproblen = len(evalprobs1)
        evalprobs1 = [prob for prob in evalprobs1 if prob != 0]  # Remove zeros from evalprobs
        
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
        for i, prob in evaldata1.items():
            evalbins.add_sample(i, prob, evaloccur1[i])
        
        # check for collisions
        max_coll = 0; b_id = None
        for (bucket1, bucket2) in zip(sampbins.buckets.values(), evalbins.buckets.values()):
            assert bucket1.id == bucket2.id
            if max_coll < check_collisions(bucket1, bucket2):
                b_id = bucket2.id
            max_coll = max(max_coll, check_collisions(bucket1, bucket2))
        if verb: print(f'max bucket ids [{b_id}] ->', max_coll)

        sampbins.normalize()
        evalbins.normalize()

        # low bucket test
        lowbuck = abs(sampbins.get_lowestBucket() - evalbins.get_lowestBucket())
        if verb: print(f"lb: {os.path.basename(evalfile).strip()} ---> {lowbuck}")

        #DKW test on the bucket distribution
        dkweval = dkwtest(sampbins, evalbins)
        if verb: print(f"dkw: {os.path.basename(evalfile).strip()} ---> {dkweval}")

        result1 = dkweval * 10000 + max_coll
        
        # identifies student generated text
        evalprobs2 = evaldata2.values()    
        earlyevalproblen = len(evalprobs2)
        evalprobs2 = [prob for prob in evalprobs2 if prob != 0]  # Remove zeros from evalprobs
        
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
        for i, prob in evaldata2.items():
            evalbins.add_sample(i, prob, evaloccur2[i])
        
        # check for collisions
        max_coll = 0; b_id = None
        for (bucket1, bucket2) in zip(sampbins.buckets.values(), evalbins.buckets.values()):
            assert bucket1.id == bucket2.id
            if max_coll < check_collisions(bucket1, bucket2):
                b_id = bucket2.id
            max_coll = max(max_coll, check_collisions(bucket1, bucket2))
            # print(f'bucket ids [{bucket2.id}] ->', check_collisions(bucket1, bucket2))
        if verb: print(f'max bucket ids [{b_id}] ->', max_coll)

        sampbins.normalize()
        evalbins.normalize()

        # low bucket test
        lowbuck = abs(sampbins.get_lowestBucket() - evalbins.get_lowestBucket())
        if verb: print(f"lb: {os.path.basename(evalfile).strip()} ---> {lowbuck}")

        #DKW test on the bucket distribution
        dkweval = dkwtest(sampbins, evalbins)
        if verb: print(f"dkw: {os.path.basename(evalfile).strip()} ---> {dkweval}")

        result2 = dkweval * 10000 + max_coll

        allpassres.append(result1)
        allflagres.append(result2)

        if debug:
            fig = plt.figure()

            bins = evalbins
            if verb: print(evalfile)
            for i in sorted(bins.keys()):
                if verb: print(i, ":", bins[i])

            plt.bar(list(bins.keys()),list(bins.values()), width = 1, alpha = 0.6)
            bins = sampbins
            plt.bar(list(bins.keys()),list(bins.values()), width = 1, alpha = 0.6)
                    # , label = lm)

            plt.xlabel(f"probability bins (in $log_{bucketwdth}$)")
            plt.ylabel("# of strings")
            # plt.legend()
            plt.ylim([0,70])
            plt.xlim([-nbucket,0])
            plt.savefig("bucket-graphs/" + evalfile.split("/")[-1].strip() + ".png")

    fpr, tpr, thresh = roc_curve([1] * len(allpassres) + [0] * len(allflagres), allpassres + allflagres)
    roc_auc = auc(fpr, tpr)
    if verb:
        print(thresh)
        print(allpassres, allflagres)
        print(f"roc_auc: {roc_auc}")
