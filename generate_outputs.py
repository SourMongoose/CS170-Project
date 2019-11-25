import solver
import os

alg = solver.NaiveSolver

# create output directory
try:
    os.mkdir("outputs")
except:
    pass

bad = set([19,47,59,69,80,91,92,119,129,138,143,158,159,176,193,198,215,221,225,243,257,270,280,283,292,302,308,310,312,313,317,323,328,330,333,336,338,339,341,344,346,347,348,349,353,354,355])

for i in range(1,367):
    if i in bad:
        continue
    if i%10==0:
        print(i)
    for j in [50,100,200]:
        try:
            g1 = alg('inputs/{0}_{1}.in'.format(i,j))
            g1.solve('outputs/{0}_{1}.out'.format(i, j))
        except:
            pass
            #print(i,j)
