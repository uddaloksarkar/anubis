from collections import defaultdict, Counter
from human_eval.human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.human_eval.execution import check_correctness
import tqdm
import contextlib
import faulthandler
import platform
import signal
import tempfile
from typing import Optional, Callable, Dict
import os
import sys
import os
import json

# Hardcoded set of keywords
stop_sequences = ["\n#","\nprint", "\nif", "\nclass", "\ndef"]

# load prompts 
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

# Function to truncate text at the first occurrence of any keyword in the set
def truncate_on_stop_sequences(input_folder, input_file, out_f, prompt, prompt_id): 
    try:
        with open(input_file, 'r') as f:
            text = f.read()
        
        if not text.startswith(prompt):
            raise NameError("Prompt and generation do not match.")    
        else:
            text = text[len(prompt):]

        # Find the first occurrence of any keyword
        min_idx = len(text)
        for kw in stop_sequences:
            idx = text.find(kw)
            if idx != -1 and idx < min_idx:
                min_idx = idx
        
        # Truncate the text at the first found keyword
        truncated_text = text[:min_idx] if min_idx != len(text) else text

        whole_text = prompt + truncated_text.rstrip() 
        
        file_data = {"task_id": f"HumanEval/{prompt_id}", "completion": whole_text}
        json.dump(file_data, out_f)
        out_f.write('\n')   
     
    except Exception as e:
        raise NameError(f"Error in the file {input_file} : {e}")


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class TimeoutException(Exception):
    pass

def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    result = []
        # Construct the check program and run it.
    check_program = (
        problem["prompt"] + completion + "\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})"
    )

    try:
        exec_globals = {}
        with time_limit(timeout):
            exec(check_program, exec_globals)
        result.append("passed")
    except TimeoutException:
        result.append("timed out")
    except BaseException as e:
        result.append(f"failed: {e}")


    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )



if __name__ == "__main__":
    import argparse
         
    parser = argparse.ArgumentParser()
    # parser.add_argument("completions", help="The  json file with the (truncated) programs")
    parser.add_argument('--inputdir', type=str,
                        dest='input_folder', help="input directory containing the text")
    parser.add_argument('--promptsrc', type=str,
                        default='HumanEval.jsonl', dest='prompts', help="prompt source")
    parser.add_argument('--outputdir', type=str,
                        dest='outputdir', help="Output directory for the results")
    parser.add_argument("--promptID", type=int, default=None, help="Indicate the prompt ID to be checked (inputfolder should be the folder containing the completions for the promptID)")
    args = parser.parse_args()

    os.system(f'mkdir -p {args.outputdir}')
    problems = read_problems(args.prompts)
    completion_id = Counter()
    n_samples = 0
    results = defaultdict(list)
    timeout = 3.0
    
    allprompts,_ = loadData('HumanEval.jsonl')
    tmpfile = f'tmp{args.promptID}.jsonl'
    prompt = problems[f'HumanEval/{args.promptID}']['prompt']

    with open(tmpfile,'w') as out_f:
        for root, dirs, files in os.walk(args.input_folder, followlinks=True):
            for filename in files:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    print(file_path)
                    truncate_on_stop_sequences(args.input_folder, file_path, out_f, prompt, args.promptID)

    futures = []
    folderprefix = f'{args.outputdir}/samples0{args.promptID}.ds'
    print("Reading samples...")
    for sample in tqdm.tqdm(stream_jsonl(tmpfile)):
        task_id = sample["task_id"]
        if args.promptID is not None and task_id.split("/")[-1] != str(args.promptID):
            continue
        completion = sample["completion"]
        future = check_correctness(problems[task_id], completion, timeout, completion_id[task_id])
        futures.append(future)
        completion_id[task_id] += 1
        n_samples += 1
        os.system(f"mkdir -p {folderprefix}")
        # if future['passed']:
        with open(f"{folderprefix}/{completion_id[task_id]-1}.py", 'w') as f: #hard coded -1 to get the correct correspondence with sample ids 
            f.write(completion)

    # assert len(completion_id) == len(problems), "Some problems are not attempted."

    results = []
    tot_passed = 0
    print("Running test suites...")
    for future in tqdm.tqdm(futures, total=len(futures)):
        result = future['result']
        results.append(result)
        if result == 'passed': tot_passed += 1
    
    print(f'total number of programs that passed the test {tot_passed} out of {n_samples}')
    os.system(f'rm {tmpfile}')
