import random
import numpy as np
from tspy import TSP
from tspy.solvers import TwoOpt_solver, NN_solver
import networkx as nx

class Solver:
    INF = 1e18

    def __init__(self, inputfile):
        with open(inputfile, 'r') as f:
            self.L = int(f.readline())
            self.H = int(f.readline())
            self.names = list(f.readline().strip().split())
            self.homes = list(f.readline().strip().split())
            self.homes = [self.names.index(h) for h in self.homes]
            self.start = self.names.index(f.readline().strip())
            self.adj = []
            for i in range(self.L):
                self.adj.append(list(f.readline().strip().split()))
                self.adj[i] = [(Solver.INF if l == 'x' else round(float(l)*1e5)) for l in self.adj[i]]

    # calculate shortest paths between any two vertices
    def FloydWarshall(self):
        self.dist = [[0]*self.L for _ in range(self.L)]
        self.path = [[[] for _ in range(self.L)] for __ in range(self.L)]
        for i in range(self.L):
            for j in range(self.L):
                self.dist[i][j] = self.adj[i][j]
                self.path[i][j] = [i,j]
        for i in range(self.L):
            self.dist[i][i] = 0
            self.path[i][i] = [i]

        for k in range(self.L):
            for i in range(self.L):
                for j in range(self.L):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                        self.path[i][j] = self.path[i][k] + self.path[k][j][1:]

    # return MST
    def Prims(self):
        edges = []

        N = self.H + (1 if self.start not in self.homes else 0)

        dist = [Solver.INF]*self.L
        visited = [False]*self.L
        prev = [-1]*self.L

        dist[self.start] = 0

        while True:
            closest = -1
            for i in self.homes + [self.start]:
                if not visited[i] and (closest == -1 or dist[i] < dist[closest]):
                    closest = i

            if closest == -1: break

            visited[closest] = True

            for i in self.homes:
                if not visited[i]:
                    newDist = self.dist[closest][i]
                    if newDist < dist[i]:
                        dist[i] = newDist
                        prev[i] = closest

        for i in self.homes:
            edges.append((i,prev[i]))
        return edges

    # calculate dropoff locations of all TAs, and total energy cost
    def calcDropoffsAndDist(self, path):
        dropoffs = {}
        for h in self.homes:
            drop = min(path, key=lambda l: self.dist[l][h])
            if drop not in dropoffs:
                dropoffs[drop] = []
            dropoffs[drop].append(h)

        totalDist = 0
        for i in range(len(path) - 1):
            totalDist += self.dist[path[i]][path[i+1]] * 2/3
        for d in dropoffs:
            for h in dropoffs[d]:
                totalDist += self.dist[d][h]

        return dropoffs, totalDist

    # compress cycles that are suboptimal on given path
    # note: prioritizes smaller cycles
    def compressCycles(self, path):
        change = True
        while change:
            change = False
            L = len(path)
            dropoffs = self.calcDropoffsAndDist(path)[0]

            for l in range(2,L):
                for i in range(L-l):
                    j = i + l
                    # cycle detected
                    if path[i] == path[j]:
                        # calculate distances
                        drive = self.dist[path[i]][path[i+1]]
                        walk = 0
                        dropped = []
                        checked = set()
                        for k in range(i+1,j):
                            drive += self.dist[path[k]][path[k+1]]
                            if path[k] in dropoffs and path[k] not in checked:
                                for h in dropoffs[path[k]]:
                                    walk += self.dist[path[k]][h]
                                    dropped.append(h)
                                checked.add(path[k])

                        # energy needed while taking cycle
                        eyes = drive * 2/3 + walk
                        # energy needed without cycle
                        eno = sum(self.dist[path[i]][h] for h in dropped)

                        if eno < eyes:
                            path = path[:i] + path[j:]
                            change = True
                            break
                if change:
                    break

        return path

    # drive on edges that more than 1 TA take
    def replaceWalkedEdges(self, path):
        # find longest common prefix
        def LCP(x, y):
            i = 0
            while len(y) > i < len(x) and x[i] == y[i]:
                i += 1
            return x[:i]

        change = True
        while change:
            change = False
            L = len(path)
            dropoffs = self.calcDropoffsAndDist(path)[0]
            checked = set()

            for i in range(L):
                p = path[i]
                # check if multiple TAs are dropped off
                if p not in checked and p in dropoffs and len(dropoffs[p]) > 1:
                    checked.add(p)
                    paths = [self.path[p][h] for h in dropoffs[p]]
                    lcp = [p]
                    for x in paths:
                        for y in paths:
                            if x != y:
                                p = LCP(x,y)
                                if len(p) > len(lcp):
                                    lcp = p
                    if len(lcp) > 1:
                        path = path[:i] + lcp + lcp[::-1][1:] + path[i+1:]
                        change = True

                if change:
                    break

        return path

    # try to remove inefficient dropoff locations
    def removeDropoffLocations(self, path):
        change = True
        while change:
            change = False
            L = len(path)
            dropoffs, dist = self.calcDropoffsAndDist(path)

            # list of dropoff locations
            locs = [[path[i],i] for i in range(L) if (path[i] in dropoffs or i in [0,L-1])]

            for i in range(1,len(locs)-1):
                prev = locs[i-1]
                next = locs[i+1]
                newpath = path[:prev[1]] + self.path[prev[0]][next[0]] + path[next[1]+1:]
                if self.calcDropoffsAndDist(newpath)[1] < dist:
                    path = newpath
                    change = True
                    break

        return path

    def solve(self):
        # to be implemented
        pass

    def output(self, outputfile='output.out', output=True):
        if output:
            # calculate dropoff locations
            dropoffs, _ = self.calcDropoffsAndDist(self.finalPath)

            with open(outputfile, 'w') as f:
                f.write(' '.join(self.names[i] for i in self.finalPath) + '\n')
                f.write(str(len(dropoffs.keys())) + '\n')
                for k in dropoffs.keys():
                    f.write(self.names[k] + ' ')
                    f.write(' '.join(self.names[i] for i in dropoffs[k]) + '\n')

    def solveAndOutput(self, outputfile='output.out', output=True):
        self.solve()
        self.output(outputfile, output)


class TSPYSolver(Solver):
    def findPath(self):
        pass

    def solve(self):
        self.FloydWarshall()
        tsp = TSP()
        tsp.read_mat(np.array(self.dist, dtype=np.float64))
        two_opt = TwoOpt_solver(initial_tour='NN', iter_num=1000000)
        tsp.get_approx_solution(two_opt)
        path = tsp.get_best_solution()
        # check all cycles along path to see if compression is better
        path = self.compressCycles(path)
        # try to remove dropoff locations
        path = self.removeDropoffLocations(path)
        # replace double walked edges
        #path = self.replaceWalkedEdges(path)

        self.finalPath = path

        # calculate final energy
        _, totalDist = self.calcDropoffsAndDist(path)

        return totalDist

# Solver that uses 2x MST approximation to calculate output
class NaiveSolver(Solver):
    # return path generated by running DFS on given MST
    def findPath(self, MST):
        adj = [[] for _ in range(self.L)]
        for edge in MST:
            a, b = edge[0], edge[1]
            adj[a].append(b)
            adj[b].append(a)

        v = set()
        v.add(self.start)
        route = []
        def DFS(i):
            route.append(i)
            for j in adj[i]:
                if j not in v:
                    v.add(j)
                    DFS(j)
        DFS(self.start)
        route.append(self.start)
        path = [self.start]
        for i in range(len(route)-1):
            #print(path)
            path.extend(self.path[route[i]][route[i+1]][1:])
        return path

    def solve(self):
        self.FloydWarshall()
        MST = self.Prims()
        self.finalPath = self.findPath(MST)

        # calculate final energy
        dropoffs, totalDist = self.calcDropoffsAndDist(self.finalPath)

        return totalDist

class NaiveSolverCompress(NaiveSolver):
    def solve(self):
        self.FloydWarshall()
        MST = self.Prims()
        path = self.findPath(MST)

        # check all cycles along path to see if compression is better
        path = self.compressCycles(path)
        # try to remove dropoff locations
        path = self.removeDropoffLocations(path)
        # replace double walked edges
        #path = self.replaceWalkedEdges(path)

        self.finalPath = path

        # calculate final energy
        dropoffs, totalDist = self.calcDropoffsAndDist(path)

        return totalDist

# same as NaiveSolverCompress, except randomize DFS of the MST
class NaiveSolverCompressRandom(NaiveSolverCompress):
    def findPath(self, MST):
        adj = [[] for _ in range(self.L)]
        for edge in MST:
            a, b = edge[0], edge[1]
            adj[a].append(b)
            adj[b].append(a)

        # randomize adjacency lists
        for i in range(self.L):
            random.shuffle(adj[i])

        v = set()
        v.add(self.start)
        route = []

        def DFS(i):
            route.append(i)
            for j in adj[i]:
                if j not in v:
                    v.add(j)
                    DFS(j)

        DFS(self.start)
        route.append(self.start)
        path = [self.start]
        for i in range(len(route) - 1):
            path.extend(self.path[route[i]][route[i + 1]][1:])
        return path

# solver that creates multiple solvers and returns the best one
class NaiveSolverMultiple:
    ranomized_algs = [NaiveSolverCompressRandom]
    deterministic_algs = [TSPYSolver]

    def __init__(self, inputfile, count=5):
        self.solvers = []
        for alg in self.ranomized_algs:
            self.solvers.extend([alg(inputfile) for _ in range(count)])
        for alg in self.deterministic_algs:
            self.solvers.append(alg(inputfile))

    def solve(self):
        print("Beginning solve...")
        dists = [s.solve() for s in self.solvers]
        self.best = self.solvers[dists.index(min(dists))]
        if isinstance(self.best, TSPYSolver):
            print("WOOHOO")
        else:
            print("DONE")
        return min(dists)

    def output(self, outputfile='output.out', output=True):
        if output:
            self.best.output(outputfile, output)

    def solveAndOutput(self, outputfile='output.out', output=True):
        self.solve()
        self.output(outputfile, output)

# Solver that uses Christofides' algorithm to calculate output
# Note: minimum matching must be optimal to achieve 1.5x bound
class ChristofidesSolver(Solver):
    # return all vertices of an MST with odd degree
    def oddVertices(self, MST):
        deg = [0] * self.L
        for edge in MST:
            a, b = edge[0], edge[1]
            deg[a] += 1
            deg[b] += 1

        odd = []
        for i in range(self.L):
            if deg[i] % 2 == 1:
                odd.append(i)

        return odd

    # return an minimum matching between odd-degree vertices of given MST
    def minMatching(self, MST):
        odd = self.oddVertices(MST)

        G = nx.Graph()
        G.add_nodes_from(odd)

        maxE = 0
        for u in odd:
            for v in odd:
                maxE = max(maxE, self.dist[u][v])

        for i in range(len(odd)):
            for j in range(i+1, len(odd)):
                G.add_edge(odd[i],odd[j],weight=maxE-self.dist[odd[i]][odd[j]])

        mm = nx.algorithms.max_weight_matching(G)

        return list(mm)

    # return an Eulerian tour of given graph G
    def EulerTour(self, G):
        adj = [set() for _ in range(self.L)]
        for edge in G:
            a, b = edge[0], edge[1]
            adj[a].add(b)
            adj[b].add(a)

        def connected(start):
            v = set()
            v.add(start)
            def dfs(i):
                for e in adj[i]:
                    if e not in v:
                        v.add(e)
                        dfs(e)
            dfs(start)
            for i in range(self.L):
                if len(adj[i]) and i not in v:
                    return False
            return True

        tour = [self.start]
        while True:
            stop = True
            edges = list(adj[tour[-1]])
            for b in edges:
                adj[tour[-1]].remove(b)
                if not connected(b):
                    adj[tour[-1]].add(b)
                else:
                    tour.append(b)
                    stop = False
                    break
            if stop:
                break

        return tour

    # return path generated by Eulerian tour
    def findPath(self, tour):
        updated = []
        for x in tour:
            if x not in updated:
                updated.append(x)
        updated.append(tour[-1])
        path = [updated[0]]
        for i in range(len(updated)-1):
            path.extend(self.path[updated[i]][updated[i+1]][1:])
        return path

    def solve(self):
        self.FloydWarshall()
        MST = self.Prims()
        MM = self.minMatching(MST)
        ET = self.EulerTour(MST+MM)
        path = self.findPath(ET)

        self.finalPath = path

        dropoffs, totalDist = self.calcDropoffsAndDist(self.finalPath)

        return totalDist

# same as above, but uses cycle and dropoff compression
class ChristofidesSolverCompress(ChristofidesSolver):
    def solve(self):
        self.FloydWarshall()
        MST = self.Prims()
        MM = self.minMatching(MST)
        ET = self.EulerTour(MST+MM)
        path = self.findPath(ET)

        # check all cycles along path to see if compression is better
        path = self.compressCycles(path)
        # try to remove dropoff locations
        path = self.removeDropoffLocations(path)

        self.finalPath = path

        dropoffs, totalDist = self.calcDropoffsAndDist(self.finalPath)

        return totalDist

