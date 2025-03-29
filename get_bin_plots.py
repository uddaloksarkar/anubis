import os
from tempfile import gettempdir
from gmpy2 import mpfr
import argparse  
from  math import log10 as _log, floor, ceil
from matplotlib import pyplot as plt
from operator import itemgetter
import random
import json
import numpy as np

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

    def add_sample(self, sample, prob, occur, bucketwdth):
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
    

def get_nbuckets(samplprobs, thresh, bucketwdth):
    samplprobsorted = sorted(samplprobs)
    maxprob = samplprobsorted[int(thresh * len(samplprobsorted))]
    # get the number of buckets needed for the rest of elems 
    try:
        nbucket = - floor(_log(maxprob) / _log(bucketwdth)) + 1
    except ValueError:
        nbucket = 1
    print("number of bucket:", nbucket)
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

def replace_percentage(A, B, A1, B1, percentage):

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

    len_B = len(B)
    len_A = len(A)
    
    num_replacements = ceil((percentage / 100) * len_B)

    # Get a list of keys in B and A
    keys_B = list(B.keys())
    keys_A = list(A.keys())

    # Randomly choose keys from B to replace
    keys_to_replace_in_B = random.sample(list(range(len_B)), num_replacements)
    keys_to_keep_in_B = list(set(list(range(len_B))) - set(keys_to_replace_in_B))

    assert len(A) == len(A1)
    assert len(B) == len(B1)
    assert len(A) == len(B)
    assert len(A1) == len(B1)

    newB, newB1 = {}, {}
    # Replace the randomly chosen keys in B
    for i in keys_to_replace_in_B:
        if i < len(keys_A):  # Replace with values from A, using A's keys
            key_A = keys_A[i]
            newB[key_A] = A[key_A]  # Replace value in B with corresponding value from A
            newB1[key_A] = A1[key_A]  # Replace value in B with corresponding value from A
        else:
            assert False
            continue  # Stop if we've run out of keys in A to replace with
    for i in keys_to_keep_in_B:
        if i < len(keys_B):  
            key_B = keys_B[i]
            newB[key_B] = B[key_B]  
            newB1[key_B] = B1[key_B]  
        else:
            assert False

    # assert len(A) == len(A1)
    # assert len(newB) == len(newB1)
    # assert len(A) == len(newB)
    # assert len(A1) == len(newB1)


    return newB, newB1


##############################
# Loading the data
##############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int,
                            default=0, help="2 = gpt, 1 = deepseek, 0 = stability",dest='model')
    parser.add_argument('--evalmodel', type=int,
                            default=1, help="2 = gpt, 1 = deepseek, 0 = stability",dest='evalmodel')
    parser.add_argument('--debug', type=int,
                            default=0, help="1 for Debugging, 0 else", dest='debug')
    parser.add_argument('--verbose', type=int,
                            default=0, help="1 for verbosity, 0 else", dest='verbose')
    parser.add_argument('--bwidth', type=int,
                            default=2, help="bucket width", dest='bwidth')
    parser.add_argument('--thresh', type=float,
                            default=0.05, help="lower bucket threshold in percentile", dest='thresh')
    parser.add_argument('--sampthresh', type=int, default=500, help="minimum number of samples", dest='sampthresh')
    parser.add_argument('--studir', type=str,dest='studir')
    parser.add_argument('--llmdir', type=str,dest='llmdir')

    args = parser.parse_args()
    model = args.model
    evalmodel = args.evalmodel
    debug = args.debug
    verb = args.verbose
    bucketwdth = args.bwidth
    thresh = args.thresh
    studir = args.studir
    llmdir = args.llmdir

    print("Loading the data ...")
    if model == 0:
        modelname = "stability"
        modelid = 'stabilityai/stable-code-3b'
    elif model == 1:
        modelname = "deepseek"
        modelid = 'deepseek-ai/deepseek-coder-1.3b-base'
    else:
        raise NotImplementedError
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
    srcdir1 = os.path.join(parentdir, studir)
    srcdir2 = os.path.join(parentdir, llmdir)
    dir1_name = os.path.basename(studir)
    dir2_name = os.path.basename(llmdir)
    outfile = f'test-{dir1_name}-{dir2_name}--lowbucktest'
    outfp = open(outfile, 'w')
    outfiledkw = f'test-{dir1_name}-{dir2_name}--dkwtest'
    outfp2 = open(outfiledkw, 'w')
    outfilebuck = f'test-{dir1_name}-{dir2_name}--bucktest'
    outfp3 = open(outfilebuck, 'w')

    cmd = f"find {os.path.join(srcdir1, 'eval*.ds_' + evalmodelname + '.json')} > {tmpfile}"
    os.system(cmd)
    with open(tmpfile) as tmp:
        allevalFiles = tmp.readlines()

    buckls, dkwls = [], []

    plag = 0; tot = 0
    ##############################
    # start bucketing
    ##############################
    for evalfile in allevalFiles:
        
        # student eval,occur files
        filepos = evalfile.strip()
        with open(filepos) as fp:
            evaldata = json.load(fp)
        fileoccur = os.path.basename(filepos).replace("eval", "occur")
        fileoccur = os.path.join(srcdir1, fileoccur)
        with open(fileoccur) as fp:
            evaloccur = json.load(fp)

        # llm eval,occur files
        sampfilepos = filepos.replace(srcdir1, srcdir2)
        try:
            with open(sampfilepos) as fp:
                sampldata = json.load(fp)        
        except FileNotFoundError:
            # if the corresponding file doesnot exist move to next
            continue
        fileoccur = os.path.basename(sampfilepos).replace("eval", "occur")
        fileoccur = os.path.join(srcdir2, fileoccur)
        with open(fileoccur) as fp:
            samploccur = json.load(fp)

        # replace some student generated by llm
        # evaldata, evaloccur = replace_percentage(sampldata, evaldata, samploccur, evaloccur, percentage)
        # assert evaldata == sampldata

        # identifies llm generated text
        samplprobs = sampldata.values()
        # identifies student generated text
        evalprobs = evaldata.values()    
        if len(evalprobs) == 0 or len(samplprobs) == 0 : continue
        earlyevalproblen = len(evalprobs)
        evalprobs = [prob for prob in evalprobs if prob != 0]  # Remove zeros from evalprobs
        earlysamplproblen = len(samplprobs)
        samplprobs = [prob for prob in samplprobs if prob != 0]  # Remove zeros from samplprobs
        
        # if low samples do skip
        if min(len(evalprobs), len(samplprobs)) < args.sampthresh: 
            print(filepos, f"evalprob lens falls down form {earlyevalproblen} to {len(evalprobs)}, and sampleprob len from {earlysamplproblen} to {len(samplprobs)}") 
            continue 
        print('filename:',os.path.basename(filepos))
        
        # hardcoding the prompt id
        promptID = sampfilepos.split("/")[-1].split(".")[0]
        
        # find the probability of the lowest bucket and number of buckets to bucket the rest
        try:
            nbucket = get_nbuckets(samplprobs, thresh, bucketwdth)
        except IndexError:
            continue

        # bucketing
        sampbins = bucket_list()
        evalbins = bucket_list()
        for bcks in range(nbucket):
            sampbins.add_bucket(-1 * bcks)
            evalbins.add_bucket(-1 * bcks)
        for i, prob in sampldata.items():
            sampbins.add_sample(i, prob, samploccur[i], bucketwdth)
        for i, prob in evaldata.items():
            evalbins.add_sample(i, prob, evaloccur[i], bucketwdth)
        # sampbins.normalize()
        # evalbins.normalize()

        # # sampbins = {-1 * bcks: 0 for bcks in range(nbucket)}
        # # evalbins = {-1 * bcks: 0 for bcks in range(nbucket)}
        # for prob in samplprobs:
        #     buckpos = floor(_log(prob)/_log(bucketwdth))
        #     # buckpos = find_index(sorted(sampbins.keys()), bn)
        #     if buckpos < 1-nbucket: buckpos = 1-nbucket
        #     sampbins[buckpos] += 1 / nsamp
        # for prob in evalprobs:
        #     buckpos = floor(_log(prob)/_log(bucketwdth))
        #     # buckpos = find_index(sorted(evalbins.keys()), bn)
        #     if buckpos < 1-nbucket: buckpos = 1-nbucket
        #     evalbins[buckpos] += 1 / nsamp
        # # print(sum(evalbins.values()))
        # # assert sum(evalbins.values()) == 1
        # # assert sum(sampbins.values()) == 1

        # # low bucket test
        # lowbuck = abs(sampbins[1-nbucket] - evalbins[1-nbucket])
        # print(f"lb: {os.path.basename(evalfile).strip()} ---> {lowbuck}")
        # #DKW test on the bucket distribution
        # dkweval = dkwtest(sampbins, evalbins)
        # print(f"dkw: {os.path.basename(evalfile).strip()} ---> {dkweval}")
        
        # check for collisions
        max_coll = 0; b_id = None
        for (bucket1, bucket2) in zip(sampbins.buckets.values(), evalbins.buckets.values()):
            assert bucket1.id == bucket2.id
            if max_coll < check_collisions(bucket1, bucket2):
                b_id = bucket2.id
            max_coll = max(max_coll, check_collisions(bucket1, bucket2))
            # print(f'bucket ids [{bucket2.id}] ->', check_collisions(bucket1, bucket2))
        print(f'max bucket ids [{b_id}] ->', max_coll)

        sampbins.normalize()
        evalbins.normalize()

        # low bucket test
        lowbuck = abs(sampbins.get_lowestBucket() - evalbins.get_lowestBucket())
        print(f"lb: {os.path.basename(evalfile).strip()} ---> {lowbuck}")

        #DKW test on the bucket distribution
        dkweval = dkwtest(sampbins, evalbins)
        print(f"dkw: {os.path.basename(evalfile).strip()} ---> {dkweval}")

        outfp.write(f"{promptID} : {lowbuck} \n")
        outfp2.write(f"{promptID} : {dkweval} \n")
        outfp3.write(f"{promptID} : {max_coll} \n")

        if max_coll < 30 and dkweval < 0.05:
            plag += 1
            print("FLAGGED")
        tot += 1

        dkwls.append(dkweval)
        buckls.append(nbucket)

        if debug:
            fig = plt.figure()

            bins = evalbins
            print(evalfile)
            for i in sorted(bins.keys()):
                print(i, ":", bins[i])

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

    print(f"Plagiarism Rate : {plag /  tot * 100} %  \n")

    buckls_indices = np.argsort(buckls)
    buckls_sorted = np.array(buckls)[buckls_indices]
    dkwls_sorted = np.array(dkwls)[buckls_indices]
    plt.plot(dkwls_sorted)
    plt.savefig("bucket-vs-dkw.png")

    outfp.close()
    outfp2.close()