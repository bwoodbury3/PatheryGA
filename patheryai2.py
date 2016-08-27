# Version 2

import copy
import random
import levelpuller
import sendpath
import sys
import os
import select

from collections import deque

mapID = 8026
mapCode = ""
mutationparameter = 5 # out of 100
tournamentSize = 5
populationSize = 200
generations = 200
breedingRange = 50 # 30 either direction

if len(sys.argv) == 2:
    mapID = sys.argv[1]

data = levelpuller.getLevel(mapID)
walls = data[0]
level = data[1]
mapcode = data[2]

checkpoints = ['s', 'a', 'b', 'c', 'd', 'e', 'f']

class Level():
    def __init__(self, level):
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

    def addBlock(self, loc): # heh, adblock
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
    def __init__(self, popSize):
        self.population = []
        self.size = popSize
        for i in range(popSize):
            ind = Individual()
            ind.generateRandomIndividual()
            self.population.append(ind)
        self.max = self.findMax()

    def evolve(self):
        newPopulation = [self.max]
        for i in range(1, self.size):
            ind1 = self.population[i]
            ind2 = self.findMate(i)
            cross = self.crossover(ind1, ind2)
            if random.randint(0, 100) < mutationparameter:
                cross.mutate()
                cross.calcFitness()
            newPopulation.append(cross)
        self.population = newPopulation
        self.max = self.findMax()
        return self.max

    def findMate(self, i):
        low = i - breedingRange
        high = i + breedingRange
        if high >= self.size:
            high = self.size - 1
        # 5 possible mates
        bestFitness = 0
        bestMate = None
        for i in range(0, 5):
            poss = random.randint(low, high)
            if self.population[poss].getFitness() > bestFitness:
                bestMate = self.population[poss]
        return bestMate

    def crossover(self, ind1, ind2):
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

    def findMax(self):
        best = self.population[0]
        for i in range(1, self.size):
            if self.population[i] > best:
                best = self.population[i]
        return best

base = Level(level)
numCheckpoints = base.numCheckpoints()

population = Population(populationSize)
for i in range(generations):
    best = population.evolve()
    print best
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = raw_input()
        break
best = population.max
print best

levelcopy = copy.deepcopy(level)
tempLevel = Level(levelcopy)
tempLevel.plotPoints(best.genes)
solStr = tempLevel.generateSolutionString(best.genes)
print(tempLevel)
sendpath.sendPath(mapcode, mapID, solStr)