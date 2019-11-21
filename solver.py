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

        dist = [Solver.INF]*self.L
        visited = [False]*self.L
        prev = [-1]*self.L

        dist[self.start] = 0

        while True:
            closest = -1
            for i in range(self.L):
                if not visited[i] and (closest == -1 or dist[i] < dist[closest]):
                    closest = i

            if closest == -1: break

            visited[closest] = True

            for i in range(self.L):
                if not visited[i]:
                    newDist = self.dist[closest][i]
                    if newDist < dist[i]:
                        dist[i] = newDist
                        prev[i] = closest

        for i in range(1,self.L):
            edges.append((i,prev[i]))
        return edges

    def solve(self, outputfile):
        # to be implemented
        pass


class NaiveSolver(Solver):
    def solve(self, outputfile):
        self.FloydWarshall()
        MST = self.Prims()

