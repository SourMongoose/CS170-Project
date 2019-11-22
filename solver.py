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

    def solve(self, outputfile='output.out', output=True):
        self.FloydWarshall()
        MST = self.Prims()
        path = self.findPath(MST)

        dropoffs, totalDist = self.calcDropoffsAndDist(path)

        if output:
            with open(outputfile, 'w') as f:
                f.write(' '.join(self.names[i] for i in path)+'\n')
                f.write(str(len(dropoffs.keys()))+'\n')
                for k in dropoffs.keys():
                    f.write(self.names[k]+' ')
                    f.write(' '.join(self.names[i] for i in dropoffs[k])+'\n')

        return totalDist


class GreedyCristofidesSolver(Solver):
    def minMatching(self, MST):
        deg = [0]*self.L
        for edge in MST:
            a, b = edge[0], edge[1]
            deg[a] += 1
            deg[b] += 1

        odd = []
        for i in range(self.L):
            if deg[i] % 2 == 1:
                odd.append(i)

        matching = []
        unmatched = odd[:]
        while unmatched:
            mini = unmatched[0]
            minj = unmatched[1]
            for i in range(len(unmatched)):
                for j in range(i+1,len(unmatched)):
                    if self.dist[unmatched[i]][unmatched[j]] < self.dist[mini][minj]:
                        mini = unmatched[i]
                        minj = unmatched[j]
            #print(unmatched)
            unmatched.remove(mini)
            unmatched.remove(minj)
            matching.append((mini,minj))

        return matching

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

    def solve(self, outputfile='output.out', output=True):
        self.FloydWarshall()
        MST = self.Prims()
        MM = self.minMatching(MST)
        ET = self.EulerTour(MST+MM)
        path = self.findPath(ET)

        dropoffs, totalDist = self.calcDropoffsAndDist(path)

        if output:
            with open(outputfile, 'w') as f:
                f.write(' '.join(self.names[i] for i in path) + '\n')
                f.write(str(len(dropoffs.keys())) + '\n')
                for k in dropoffs.keys():
                    f.write(self.names[k] + ' ')
                    f.write(' '.join(self.names[i] for i in dropoffs[k]) + '\n')

        return totalDist