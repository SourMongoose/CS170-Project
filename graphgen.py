import random
import math

class DisjointSets:
    def __init__(self, N):
        self.count = N
        self.parent = list(range(N))
        self.weight = [1]*N

    def root(self, p):
        while p != self.parent[p]:
            p = self.parent[p]
        return p

    def connect(self, p, q):
        rP, rQ = self.root(p), self.root(q)
        if rP == rQ: return

        if self.weight[rP] < self.weight[rQ]:
            self.parent[rP] = rQ
            self.weight[rQ] += self.weight[rP]
        else:
            self.parent[rQ] = rP
            self.weight[rP] += self.weight[rQ]

        self.count -= 1

    def connected(self, p, q):
        return self.root(p) == self.root(q)

class Graph:
    def __init__(self, L, H, E, mx=1e14):
        self.L = L
        self.H = H
        self.start = 0

        self.names = [str(i) for i in range(L)]
        self.homes = [str(i) for i in range(H)]
        random.shuffle(self.names)
        random.shuffle(self.homes)

        locations = []
        while len(locations) < L:
            v = (random.randint(0,mx), random.randint(0,mx))
            if v not in locations:
                locations.append(v)

        def dist(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        self.adj = []
        for _ in range(L): self.adj.append([0]*L)
        ds = DisjointSets(L)
        edges = 0
        while edges < E or ds.count > 1:
            a = random.randint(0,L-1)
            b = random.randint(0,L-1)
            if a == b: continue
            if self.adj[a][b] > 0: continue

            self.adj[a][b] = self.adj[b][a] = round(dist(locations[a], locations[b])) / 1e5
            edges += 1
            ds.connect(a, b)

    def output_graph(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self.L)+'\n')
            f.write(str(self.H)+'\n')
            f.write(' '.join(self.names)+'\n')
            f.write(' '.join(self.homes)+'\n')
            f.write(str(self.start)+'\n')
            for row in self.adj:
                f.write(' '.join((str(l) if l > 0 else 'x') for l in row)+'\n')

