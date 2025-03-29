import subprocess
import re
import os, json
import argparse
import datetime

date = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int,
                        default=0, dest='offset')   # offset to add to the rank
args = parser.parse_args()

# process rank

rank = (os.environ.get('OMPI_COMM_WORLD_RANK') or
        os.environ.get('PMI_RANK') or
        os.environ.get('I_MPI_RANK'))

# rank=0
if rank is not None:
    rank = int(rank)
else:
    rank = 0

ntask = 20   # number of task per machine
avgpred = {}    # store the average prediction for each task

for task in range(ntask):
    curr_task = rank*ntask + task + int(args.offset)
    print(curr_task)
    result = subprocess.run(
        ['python3', 'run.py', '--output_name', 'main', '--base_model_name', 'google/codegemma-2b', '--n_perturbation_list', '100', '--n_samples', str(100), '--pct_words_masked', str(0.3), '--span_length', str(2), '--batch_size', str(10), '--skip_baselines', '--task', str(curr_task), '--dataset', 'codegemma1', '--do_top_p'],
        capture_output=True, text=True
    )

#     match = re.search(r'Average real preds:\s*([^\s%|]+)', result.stdout)
#     if match:
#         avgpred[curr_task] = float(match.group(1))
#     else:
#         print("Error: No match found")
#         print(result.stdout)
#         print(result.stderr)

# # Print collected Success Rates
# print(date.strftime("%d"))
# with open(f"deepseek-evaluation-{rank+int(args.offset)}.json", "w") as file:
#     json.dump(avgpred, file, indent=4)
        