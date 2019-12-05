import solver
import os
import sys, traceback

alg = solver.NaiveSolverMultiple

# create output directory
try:
    os.mkdir("outputs")
except:
    pass

parallelize = True

bad = set([19,47,59,69,80,91,92,119,129,138,143,158,159,176,193,198,215,221,225,243,257,270,280,283,292,302,308,310,312,313,317,323,328,330,333,336,338,339,341,344,346,347,348,349,353,354,355])

def handle_err(e, i):
    print("Exception occurred on input {0}:".format(i))
    print("-"*60)
    traceback.print_exc(file=sys.stdout)
    print("-"*60)

def solve(i):
    if i%10==0:
        print(i)
    if i in bad:
        return

    try:
        g1 = alg('inputs/{0}_50.in'.format(i))
        g1.solveAndOutput('outputs/{0}_50.out'.format(i))
    except Exception as e:
        handle_err(e, i)

    try:
        g1 = alg('inputs/{0}_100.in'.format(i))
        g1.solveAndOutput('outputs/{0}_100.out'.format(i))
    except Exception as e:
        handle_err(e, i)

    try:
        g1 = alg('inputs/{0}_200.in'.format(i))
        g1.solveAndOutput('outputs/{0}_200.out'.format(i))
    except Exception as e:
        handle_err(e, i)

if parallelize:
    import multiprocessing as mp
    num_cpu = mp.cpu_count()
    with mp.Pool(processes=num_cpu) as p:
        p.map(solve, list(range(1, 367)))
else:
    for i in range(1, 367):
        solve(i)
