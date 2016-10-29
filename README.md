# PatheryGA
Genetic Algorithm for solving Pathery maps (http://www.pathery.com)

Usage:
python patheryai.py [levelID]
NOTE: only works for maps without teleporters for now
      (this is because the path that my solver takes
      is not nevessarily the path that Pathery's solver takes)

Creates a population of 200 solves and breeds them together for a custom
number of generations.

Empirically, it converges to solutions 70-80% that of the top scores.

patheryai2.py experiements with a different breeding technique - breeding
with distance as a factor to prevent convergence on local maxima.
