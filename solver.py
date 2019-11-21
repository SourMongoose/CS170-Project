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

    def solve(self, outputfile):
        # to be implemented
        pass


class NaiveSolver(Solver):
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

    def solve(self, outputfile):
        self.FloydWarshall()
        MST = self.Prims()
        path = self.findPath(MST)
        dropoffs = {}
        for h in self.homes:
            drop = min(path, key=lambda l: self.dist[l][h])
            if drop not in dropoffs:
                dropoffs[drop] = []
            dropoffs[drop].append(h)

        with open(outputfile, 'w') as f:
            f.write(' '.join(self.names[i] for i in path)+'\n')
            f.write(str(len(dropoffs.keys()))+'\n')
            for k in dropoffs.keys():
                f.write(self.names[k]+' ')
                f.write(' '.join(self.names[i] for i in dropoffs[k])+'\n')
