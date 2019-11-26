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
    def compressCycles(self, path):
        change = True
        while change:
            change = False
            L = len(path)
            dropoffs = self.calcDropoffsAndDist(path)[0]
            for i in range(L):
                drive = walk = 0
                dropped = []
                checked = set()

                for j in range(i+1, L):
                    drive += self.dist[path[j-1]][path[j]]

                    # cycle detected
                    if path[i] == path[j]:
                        # energy needed while taking cycle
                        eyes = drive * 2/3 + walk
                        # energy needed without cycle
                        eno = sum(self.dist[path[i]][h] for h in dropped)

                        if eno < eyes:
                            path = path[:i] + path[j:]
                            change = True

                        break

                    # check for dropoffs
                    if path[j] in dropoffs and path[j] not in checked:
                        for h in dropoffs[path[j]]:
                            walk += self.dist[path[j]][h]
                            dropped.append(h)
                        checked.add(path[j])

                # for now, break once we take out a cycle.
                # note: may want to remove "worst" cycle instead (one with highest energy cost)
                if change:
                    break

        return path

    # compress cycles that are suboptimal on given path
    # note: prioritizes smaller cycles
    def compressCycles2(self, path):
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

    def solve(self, outputfile):
        # to be implemented
        pass


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

    def solve(self, outputfile='output.out', output=True):
        self.FloydWarshall()
        MST = self.Prims()
        path = self.findPath(MST)

        # calculate final dropoff locations
        dropoffs, totalDist = self.calcDropoffsAndDist(path)

        # output to file
        if output:
            with open(outputfile, 'w') as f:
                f.write(' '.join(self.names[i] for i in path)+'\n')
                f.write(str(len(dropoffs.keys()))+'\n')
                for k in dropoffs.keys():
                    f.write(self.names[k]+' ')
                    f.write(' '.join(self.names[i] for i in dropoffs[k])+'\n')

        return totalDist

class NaiveSolverCompress(NaiveSolver):
    def solve(self, outputfile='output.out', output=True):
        self.FloydWarshall()
        MST = self.Prims()
        path = self.findPath(MST)

        # check all cycles along path to see if compression is better
        path = self.compressCycles(path)

        # calculate final dropoff locations
        dropoffs, totalDist = self.calcDropoffsAndDist(path)

        # output to file
        if output:
            with open(outputfile, 'w') as f:
                f.write(' '.join(self.names[i] for i in path)+'\n')
                f.write(str(len(dropoffs.keys()))+'\n')
                for k in dropoffs.keys():
                    f.write(self.names[k]+' ')
                    f.write(' '.join(self.names[i] for i in dropoffs[k])+'\n')

        return totalDist

class NaiveSolverCompress2(NaiveSolver):
    def solve(self, outputfile='output.out', output=True):
        self.FloydWarshall()
        MST = self.Prims()
        path = self.findPath(MST)

        # check all cycles along path to see if compression is better
        path = self.compressCycles2(path)
        # replace double walked edges
        path = self.replaceWalkedEdges(path)

        # calculate final dropoff locations
        dropoffs, totalDist = self.calcDropoffsAndDist(path)

        # output to file
        if output:
            with open(outputfile, 'w') as f:
                f.write(' '.join(self.names[i] for i in path)+'\n')
                f.write(str(len(dropoffs.keys()))+'\n')
                for k in dropoffs.keys():
                    f.write(self.names[k]+' ')
                    f.write(' '.join(self.names[i] for i in dropoffs[k])+'\n')

        return totalDist

# Solver that uses Christofides' algorithm to calculate output
# Note: since minimum matching is not optimal, 1.5x bound is not achieved
class GreedyCristofidesSolver(Solver):
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

    # return an (approximate) minimum matching between odd-degree vertices of given MST
    def minMatching(self, MST):
        odd = self.oddVertices(MST)

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