import sys
import os
import tempfile
from math import log2 
import time

tempDir = tempfile.gettempdir()
rank = (os.environ.get('OMPI_COMM_WORLD_RANK') or
        os.environ.get('PMI_RANK') or
        os.environ.get('I_MPI_RANK'))

if rank is None:
    print("Not running under an MPI environment or the environment variable is not set.")
else:
    print(f"Process rank: {rank}")

folder = 'codegemma'
extension = 'starcoder'
sampdir = 'corpus-100'

cmd = f'python semantic_check.py {sampdir}/{folder}/samples0{rank}.ds_{extension}/ HumanEval.jsonl {sampdir}-sanitized/{folder}/ --promptID {rank}'
os.system(cmd)

