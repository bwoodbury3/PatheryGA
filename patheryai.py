import copy
import random
import levelpuller
import sendpath
import sys
import os
import select

from collections import deque
from sets import Set

mapID = 8026
mapCode = ""
mutationparameter = 5 # out of 100
tournamentSize = 5
populationSize = 200
generations = 200

if len(sys.argv) == 2:
    mapID = sys.argv[1]

data = levelpuller.getLevel(mapID)
walls = data[0]
level = data[1]
mapcode = data[2]

if data[3] == True:
    print "The solver does not yet account for teleporters"
    print "Please try another mapID that does not have a"
    print "  teleporter."
    sys.exit(1)


print "Walls: " + str(walls)

checkpoints = ['s', 'a', 'b', 'c', 'd', 'e', 'f'];

class Level():
    def __init__(self, level = [['s',   0,   0,   0,   0,   0,   0,   0,   0,   0, 'f'], \
                                ['s',   0,   0,   0,   0,   0,   0,   0,   0,   0, 'f'], \
                                ['s',   0,   0,   0,   0,   0,   0,   0,   0,   0, 'f'], \
                                ['s',   0,   0,   0,   0, 'a',   0,   0,   0,   0, 'f'], \
                                ['s',   0,   0,   0,   0,   0,   0,   0,   0,   0, 'f'], \
                                ['s',   0,   0,   0,   0,   0,   0,   0,   0,   0, 'f'], \
                                ['s',   0,   0,   0,   0,   0,   0,   0,   0,   0, 'f']]):
        self.level = level

    def __repr__(self):
        l = ''
        for row in self.level:
            for item in row:
                if item == 0:
                    l += "0 "
                else:
                    l += item + " "
            l += "\n"
        return l

    def solve(self):
        numRows = self.rows()
        numCols = self.cols()

        numCheckpoints = self.numCheckpoints()
        levelCheckpoints = [0] * numCheckpoints
        levelCheckpoints[-1] = 'f'
        for i in range(0, numCheckpoints - 1):
            levelCheckpoints[i] = checkpoints[i]

        score = 0
        for i in range(1, numCheckpoints):
            queue = deque()
            visited = []
            for loc in self.locations(levelCheckpoints[i]):
                queue.appendleft((loc, 0))
                visited.append(loc)
            cur = queue.pop()
            while (self.get(cur[0]) != levelCheckpoints[i - 1]):

                dist = cur[1]
                surroundings = self.surroundings(cur[0])
                for surr in surroundings:
                    if (not surr in visited) and self.get(surr) != 'x' and self.get(surr) != 'X':
                        queue.appendleft((surr, dist + 1))
                        visited.append(surr)
                if (len(queue) == 0):
                    return -1
                cur = queue.pop()
            score += cur[1]
        return score

    def getSolvePath(self):
        numRows = self.rows()
        numCols = self.cols()

        numCheckpoints = self.numCheckpoints()
        levelCheckpoints = [0] * numCheckpoints
        levelCheckpoints[-1] = 'f'
        for i in range(0, numCheckpoints - 1):
            levelCheckpoints[i] = checkpoints[i]

        path = Set()
        for i in range(1, numCheckpoints):
            queue = deque()
            visited = {}
            for loc in self.locations(levelCheckpoints[i]):
                queue.appendleft((loc, 0))
                visited[loc] = 0
            cur = queue.pop()
            while (self.get(cur[0]) != levelCheckpoints[i - 1]):
                dist = cur[1]
                surroundings = self.surroundings(cur[0])
                for surr in surroundings:
                    if (not surr in visited) and self.get(surr) != 'x' and self.get(surr) != 'X':
                        queue.appendleft((surr, dist + 1))
                        visited[surr] = dist + 1
                if (len(queue) == 0):
                    return -1
                cur = queue.pop()

            trace = cur
            while (self.get(trace[0]) != levelCheckpoints[i]):
                surroundings = self.surroundings(trace[0])
                for surr in surroundings:
                    if surr in visited and visited[surr] == trace[1] - 1:
                        trace = (surr, visited[surr])
                if self.get(trace[0]) != levelCheckpoints[i]:
                    path.add(trace[0])

        return path

    def addBlock(self, loc):
        if (self.level[loc[0]][loc[1]] == 0):
            self.level[loc[0]][loc[1]] = 'x'
            return True
        return False

    def numCheckpoints(self):
        # find out how many checkpoints there are
        seen = []
        for row in self.level:
            for letter in row:
                if not letter in seen:
                    seen.append(letter)
        if 'e' in seen:
            return 7
        if 'd' in seen:
            return 6
        if 'c' in seen:
            return 5
        if 'b' in seen:
            return 4
        if 'a' in seen:
            return 3
        else:
            return 2

    def surroundings(self, loc):
        surr = []
        if loc[0] != 0:
            surr.append((loc[0] - 1, loc[1]))
        if loc[1] != 0:
            surr.append((loc[0], loc[1] - 1))
        if loc[0] != self.rows() - 1:
            surr.append((loc[0] + 1, loc[1]))
        if loc[1] != self.cols() - 1:
            surr.append((loc[0], loc[1] + 1))
        return surr

    def plotPoints(self, points):
        for point in points:
            self.addBlock(point)

    def generateSolutionString(self, points):
        solStr = "."
        for point in points:
            solStr += str(point[0] + 1) + "," + str(point[1]) + "."
        return solStr

    def get(self, loc):
        return self.level[loc[0]][loc[1]]

    def rows(self):
        return len(self.level)

    def cols(self):
        return len(self.level[0])

    def locations(self, letter):
        locations = []
        for row in range(self.rows()):
            for col in range(self.cols()):
                if level[row][col] == letter:
                    locations.append((row, col))
        return locations

    def checkpointExists(self, letter):
        for row in self.level:
            if letter in self.row:
                return True

class Individual:
    geneLength = walls # number of blocks

    def __init__(self):
        self.fitness = 0
        self.genes = []

    def __repr__(self):
        sc = "Score: " + str(self.fitness)
        return sc

    def __cmp__(self, other):
        return self.fitness > other.fitness

    def addGene(self, gene):
        self.genes.append(gene)

    def geneExists(self, gene):
        return gene in self.genes

    # randomly generate a bunch of blocks
    def generateRandomIndividual(self):
        rows = len(level)
        cols = len(level[0])
        for i in range(Individual.geneLength):
            validPos = False
            while not validPos:
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                if level[r][c] == 0 and not ((r, c) in self.genes):
                    validPos = True
                    self.genes.append((r, c))
        self.calcFitness()

    def getSolvePath(self):
        temp = Level(copy.deepcopy(level))
        for gene in self.genes:
            temp.addBlock(gene)
        return temp.getSolvePath()

    # uses the global level variable
    def calcFitness(self):
        temp = Level(copy.deepcopy(level))
        for gene in self.genes:
            temp.addBlock(gene)
        self.fitness = temp.solve()

    def getGenes(self):
        return self.genes

    def getFitness(self):
        return self.fitness

    def mutate(self):
        rows = len(level)
        cols = len(level[0])
        randgene = random.randint(0, Individual.geneLength - 1)
        validPos = False
        while not validPos:
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            if level[r][c] == 0 and not ((r, c) in self.genes):
                validPos = True
                self.genes[randgene] = (r, c)

class Population:
    @staticmethod
    def createPopulation(popSize):
        population = []
        for i in range(popSize):
            ind = Individual()
            ind.generateRandomIndividual()
            population.append(ind)
        return sorted(population, key=lambda x: x.fitness, reverse=True)

    @staticmethod
    def evolve(population):
        size = len(population)
        newPopulation = [population[0]]
        for i in range(1, size):
            ind1 = Population.tournamentSelection(population)
            ind2 = Population.tournamentSelection(population)
            cross = Population.crossover(ind1, ind2)
            if random.randint(0, 100) < mutationparameter:
                cross.mutate()
                cross.calcFitness()
            newPopulation.append(cross)
        best = newPopulation[0]
        for individual in newPopulation:
            if individual.getFitness() > best.getFitness():
                best = individual
        newPopulation[0] = best
        return sorted(newPopulation, key=lambda x: x.fitness, reverse=True)

    @staticmethod
    def tournamentSelection(population):
        fittest = population[random.randint(0, populationSize - 1)]
        for i in range(0, tournamentSize - 1):
            x = random.randint(0, populationSize - 1)
            if population[x].getFitness() > fittest.getFitness():
                fittest = population[x]
        return fittest

    @staticmethod
    def crossover(ind1, ind2):
        newInd = Individual()
        for i in range(len(ind1.genes)):
            if random.randint(1, 100) < 50:
                if not newInd.geneExists(ind1.genes[i]):
                    newInd.addGene(ind1.genes[i])
                else:
                    newInd.addGene(ind2.genes[i])
            else:
                if not newInd.geneExists(ind2.genes[i]):
                    newInd.addGene(ind2.genes[i])
                else:
                    newInd.addGene(ind1.genes[i])
        newInd.calcFitness()
        return newInd

class Simulation:
    def __init__(self):
        self.population = Population.createPopulation(populationSize)

    def run(self):
        print "Running a simulation"
        best = 0
        noimprovement = 0
        for i in range(generations):
            self.population = Population.evolve(self.population)
            newbest = self.population[0].getFitness()
            print self.population[0]
            if newbest == best:
                noimprovement += 1
            else:
                best = newbest
                noimprovement = 0
            # This will add more variation to the mix if the population is
            # starting to converge
            if noimprovement > 10:
                self.population = self.population[0:10] + Population.createPopulation(populationSize - 10)
            if noimprovement > 30:
                print "Converged"
                i = generations
                levelcopy = copy.deepcopy(level)
                tempLevel = Level(levelcopy)
                tempLevel.plotPoints(self.population[0].genes)
                solStr = tempLevel.generateSolutionString(self.population[0].genes)
                return (self.population[0].genes, newbest)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = raw_input()
                break

    # Find the best place for a wall
    def optimize(self, genes):
        leastInfluential = self.leastInfluentialWall(genes)
        newGenes = []
        for gene in genes:
            if gene != leastInfluential[0]:
                newGenes.append(gene)
        best = self.bestPlaceForNewWall(newGenes)[0]
        newGenes.append(best)
        return newGenes

    # This finds the wall with the least impact
    def leastInfluentialWall(self, genes):
        # This is the difference between the score with and the score without
        #   this wall
        geneWeights = []
        previousIndividual = Individual()
        map(lambda g: previousIndividual.addGene(g), genes)
        previousIndividual.calcFitness()
        baselineScore = previousIndividual.getFitness()
        for gene in genes:
            individual = Individual()
            map(lambda g: individual.addGene(g) if g != gene else None, genes)
            individual.calcFitness()
            geneWeights.append((gene, individual.getFitness()))
        # Get the wall that leaves the solution with the greatest score
        least = sorted(geneWeights, key=lambda x: x[1], reverse=True)[0]
        return least

    # This finds the best place to put a wall
    def bestPlaceForNewWall(self, genes):
        previousIndividual = Individual()
        map(lambda g: previousIndividual.addGene(g), genes)
        previousIndividual.calcFitness()
        baselineScore = previousIndividual.getFitness()
        path = self.getPath(genes)
        most = ((-1, -1), -1)
        for loc in path:
            individual = Individual()
            map(lambda g: individual.addGene(g), genes)
            individual.addGene(loc)
            individual.calcFitness()
            weight = individual.getFitness()
            if weight > most[1]:
                most = (loc, weight)
        return most

    def getPath(self, genes):
        individual = Individual()
        map(lambda g: individual.addGene(g), genes)
        path = individual.getSolvePath()
        return path

def submit(genes):
    pass

eonCount = 1
optimizeCount = 5
eons = []
for i in range(eonCount):
    simulation = Simulation()
    eons.append(simulation.run())
    print eons[0][0]
    submit(eons[i][0])

eons = sorted(eons, key=lambda x: x[1], reverse=True)

print ""
print "------Before optimizing------"
temp = Level(copy.deepcopy(level))
for gene in eons[0][0]:
    temp.addBlock(gene)
print temp
print "Score: " + str(eons[0][1])

newEons = []

for eon in eons:
    newEon = None
    for i in range(optimizeCount):
        newEon = simulation.optimize(eon[0])
        individual = Individual()
        map(lambda g: individual.addGene(g), newEon)
        individual.calcFitness()
        weight = individual.getFitness()
    newEons.append((newEon, weight))

newEons = sorted(newEons, key=lambda x: x[1], reverse=True)

print "------After optimizing------"
temp = Level(copy.deepcopy(level))
for gene in newEons[0][0]:
    temp.addBlock(gene)
print temp
print "Score: " + str(newEons[0][1])
