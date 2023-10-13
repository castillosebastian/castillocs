---
weight: 1
title: "Can we imitate Nature's evolutionary abilities? "
date: 2023-10-01
lastmod: 2023-10-01
draft: false
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
- name: "featured-image-preview"
  src: "featured-image-preview.jpg"
tags: ["evolutionary algorithms", "heuristic search", "machine learning"]
categories: ["Genetic Algorithms"]
lightgallery: true
toc:
  auto: false
---

In the rich theatre of Nature, few creatures exemplify the power of evolutionary optimization as strikingly as the hummingbird. Similarly, in the field of artificial intelligence, Genetic Algorithms stand among the select few that have successfully harnessed the principles of evolution for optimization. In this post, I will discuss these techniques that solve complex problems through an evolutionary process, leading to optimal or near-optimal solutions.

<!--more-->

# GENETIC ALGORITHMS

Few creatures exemplify the power of evolutionary optimization as strikingly as the hummingbird. With its astonishing ability to hover in place and dart with unparalleled agility, this little bird serves as a living testament to the power of natural selection in fine-tuning species for specific ecological niches. Much like this remarkable creature, Genetic Algorithms (GA) stand as a testament to the ingenuity of artificial intelligence. These algorithms, part of the broader field of evolutionary computing, harness the principles of natural evolution to find optimal or near-optimal solutions to complex problems. Just as the hummingbird has evolved to become a master of aerodynamic efficiency, Genetic Algorithms evolve candidate solutions to arrive at the best possible answer, making them invaluable tools in various sectors from engineering to data science.

## Some biological context related to GA

Genetic algorithms, a subset of the field of evolutionary computing in artificial intelligence, draw their inspiration from Darwin's theory of evolution to find the most fit solutions to complex problems. The concept of evolutionary computing itself dates back to the 1960s, initiated by Rechenberg's work on 'Evolution Strategies.' It was further advanced by John Holland, who not only conceptualized Genetic Algorithms but also enriched the domain through his seminal book 'Adaption in Natural and Artificial Systems' published in 1975. Adding another layer to this, John Koza in 1992 employed genetic algorithms to evolve LISP programs for specific tasks, coining the term 'genetic programming' for his approach.

As a nature-inspired procedure, Genetic Algorithms (GA) are fundamentally built upon key biological concepts. For instance, every living organism is composed of cells, each containing a uniform set of chromosomes. These chromosomes, essentially strings of DNA, act as blueprints for the entire organism. They consist of individual genes ---blocks of DNA--- that encode specific proteins. In simpler terms, each gene governs a particular trait, such as eye color, with various possible configurations for that trait, known as alleles. Each gene occupies a designated position on the chromosome, referred to as its locus. The full structure of an organism's genetic material is called its genome, and a specific set of genes within that genome constitutes its genotype. This genotype, influenced by developments after birth, lays the foundation for the organism's phenotype, shaping both its physical and mental attributes.

Evolution chiefly involves generational changes, making reproduction another cornerstone in the understanding of GA. During natural reproduction, a process known as recombination or crossover occurs first, wherein genes from both parents merge to create a new chromosome, thus forming a new individual. This offspring may then undergo mutations, which are slight alterations in DNA elements, primarily induced by errors during the gene-copying process.

Lastly, an organism's fitness is gauged by its ability to survive and thrive throughout its lifetime, which -in the contexto of an optimization problems- means that the "fitness" of a particular solution is measured by how well it satisfies the set criteria or objectives of the problem at hand.

## The essence of human-like optimization

When we talk about optimization, whether in business decisions or algorithmic computations, the ultimate aim is to enhance performance. As Goldberg (1989) explains:

> *What are we trying to accomplish when we optimize? Optimization seeks to improve performance toward some optimal point or points. Note that this definition has two parts: 1) we seek improvement to approach some 2) optimal point. There is a clear distinction between the process of improvement and the destination or optimum itself. Yet, in judging optimization procedures, we commonly focus solely upon convergence (does the method reach the optimum?) and forget entirely about interim performance.*

These ideas are particularly crucial in Genetic Algorithms where we deal with complex search spaces comprising of all feasible solutions to a problem. Each solution can be represented as a point in this space and evaluated based on its 'fitness' or suitability to the problem at hand. Traditional perspectives on optimization often focus narrowly on convergence while overlooking the importance of intermediate performance levels. This inclination likely finds its roots in the calculus-based origins of optimization theories. However, such a viewpoint doesn't fit naturally with complex systems or real-world decision-making scenarios, such as in business. Here, the emphasis isn't necessarily on achieving a singular 'best' outcome but rather on making quality decisions within available resources. Success is often relative to competition and context, aiming for a 'satisficing' level of performance (Simon, 1969).

Applying this more nuanced understanding of optimization to Genetic Algorithms means recognizing that the journey toward the optimal is as significant as the destination itself. Within the search space, GA operates by generating new potential solutions as evolution progresses, each with its own fitness value. The optimization process thus becomes dynamic, continuously updating as new points are discovered in the search space. In complex systems, the goal isn't merely to locate an extreme value (minimum or maximum) but to progressively improve, even if reaching the 'perfect' optimum may be elusive.

In sum, when using Genetic Algorithms for optimization, the objective extends beyond mere convergence to an optimal point. The focus is also on the quality of interim solutions, which is essential for handling complex systems where 'good enough' often suffices. It's not merely about finding the best solution but about consistently striving for better ones."

## Search for solutions

In genetic algorithms, the search for a solution is carried on through an evolutionary process that begins with a collection of individuals, commonly referred to as a 'population'. Each individual in this population is represented by its own unique chromosome, which is essentially an encoded set of attributes. Initially, this population is often generated randomly, representing a diverse range of potential solutions to the problem at hand.

Members of one population are selected to create a new generation, guided by the aspiration that this new set of individuals will outperform the preceding ones. To achieve this aim, individuals are selected based on their 'fitness'---a measure of their suitability for solving the given problem. The higher their fitness, the greater their chance of being chosen to produce offspring. To quantify this fitness, we employ a fitness function (also known as an evaluation function) that takes an individual as input and returns a scalar value. This numerical output enables us to compare the fitness levels of different individuals within the population.

This evolutionary cycle continues until a certain condition is met, such as reaching a predetermined number of generations or achieving a sufficient improvement in the best solution.

**Procedure**

1.  **\[Start\]** Generate a random population of `n` chromosomes (suitable solutions for the problem)
2.  **\[Fitness\]** Evaluate the fitness `f(x)` of each chromosome `x` in the population
3.  **\[New Population\]** Create a new population by repeating the following steps until the new population is complete:
    1.  **\[Selection\]** Select two parent chromosomes from a population according to their fitness (the better the fitness, the bigger the chance to be selected).
    2.  **\[Crossover\]** With a crossover probability, cross over the parents to form new offspring (children). If no crossover was performed, the offspring is an exact copy of the parents.
    3.  **\[Mutation\]** With a mutation probability, mutate new offspring at each locus (position in chromosome).
    4.  **\[Accepting\]** Place new offspring in the new population.
4.  **\[Replace\]** Use the new generated population for a further run of the algorithm.
5.  **\[Test\]** If the end condition is satisfied, stop, and return the best solution in the current population.
6.  **\[Loop\]** Go to step 2.

## Encoding of a Chromosome

A chromosome encapsulates the information of the solution it represents. The most common form of encoding is a binary string. For instance:

-   **Chromosome 1**: `1101100100110110`
-   **Chromosome 2**: `1101111000011110`

In this binary encoding, each bit could signify certain characteristics of the solution. Alternatively, the entire string could represent a numerical value, an approach often employed in basic GA implementations. However, encoding can vary depending on the problem at hand. For example, one could use integer or real numbers, or even encode permutations.

## Crossover

Once the encoding method is chosen, the next step is the crossover operation. Crossover blends genes from parent chromosomes to produce new offspring. The simplest method involves picking a random crossover point and merging parts of the two parent chromosomes. Here's a quick illustration (where `|` indicates the crossover point):

-   **Chromosome 1**: `11011 | 00100110110`
-   **Chromosome 2**: `11010 | 11000011110`
-   **Offspring 1**: `11011 | 11000011110`
-   **Offspring 2**: `11010 | 00100110110`

Multiple crossover points can also be utilized, and the complexity of the crossover operation is often dictated by the type of encoding used.

The crossover probability dictates the frequency of crossover operations. If crossover is bypassed, the offspring become exact replicas of their parents. On the other hand, if crossover is performed, the offspring inherit traits from both parents' chromosomes.

-   **100% Crossover Probability**: All offspring are created through crossover.
-   **0% Crossover Probability**: The new generation is produced using exact copies of the chromosomes from the previous generation. Note that this doesn't necessarily mean the new generation will be identical to the old one.

The objective of crossover is to combine the advantageous traits from each parent, thereby generating improved offspring. However, it's often beneficial to allow a portion of the older population to continue into the next generation.

## Mutation

Following crossover, mutation comes into play. The purpose of mutation is to avoid convergence of the entire population to a local optimum. It involves making random changes to the offspring generated by the crossover. In the case of binary encoding, this could mean flipping random bits from 1 to 0 or vice versa. For example:

-   **Original Offspring 1**: `1101111000011110`
-   **Original Offspring 2**: `1101100100110110`
-   **Mutated Offspring 1**: `1100111000011110`
-   **Mutated Offspring 2**: `1101101100110110`

Like crossover, the specific technique used for mutation largely depends on the chosen encoding method. For instance, if permutations are being encoded, mutation could be executed by swapping two genes.

Mutation probability determines how frequently mutations will occur within a chromosome. In the absence of mutation, the offspring are produced either directly following crossover or as direct copies, with no changes applied.

-   **100% Mutation Probability**: The entire chromosome undergoes mutation.
-   **0% Mutation Probability**: No changes occur within the chromosome.

Mutation serves as a mechanism to prevent the genetic algorithm from converging to local optima. However, excessive mutation is counterproductive, as the algorithm may essentially devolve into a random search.

## Selection

As outlined early, chromosomes are selected from the population to serve as parents for the crossover operation. The challenge lies in deciding which chromosomes to select. Darwin's theory of evolution suggests that the fittest individuals are more likely to survive and produce offspring. There are several methods for selecting these "fit" chromosomes, such as roulette wheel selection, Boltzmann selection, tournament selection, rank selection, steady-state selection, and others. This sections will describe some of these methods.

**Roulette Wheel Selection**

In this method, parents are selected based on their fitness levels. The fitter the chromosome, the higher the chance it has of being selected. Imagine a roulette wheel where each section is proportional to the fitness value of a chromosome. A marble is rolled on this wheel, and the chromosome where it stops is selected. Essentially, chromosomes with higher fitness values have larger sections and are more likely to be chosen.

Here's how the algorithm works:

1.  **\[Sum\]** Calculate the sum `( S )` of all chromosome fitnesses in the population.
2.  **\[Select\]** Generate a random number `( r )` from the interval `( (0, S) )`.
3.  **\[Loop\]** Iterate through the population, summing fitnesses from 0 to `( s )`. Stop and return the chromosome when `( s > r )`.

**Rank Selection**

Roulette wheel selection may become problematic when there are large disparities in fitness values. In such cases, rank selection can be more appropriate. In this method, chromosomes are first ranked. Each chromosome then receives a fitness score based on this ranking, from 1 (least fit) to ( N ) (most fit), where ( N ) is the number of chromosomes in the population.

**Steady-State Selection**

This isn't a specific method of selecting parents, but rather an approach to population management. In this model, a large portion of the existing chromosomes can survive to the next generation. The basic idea is to select a few good chromosomes for creating new offspring, remove some less-fit chromosomes, and place the new offspring in their spots.

**Elitism**

Elitism is an approach where the best chromosome(s) are directly transferred to the next generation to ensure that the optimal solutions found so far are not lost. This can significantly improve the performance of a GA.

# Implementing GA

This example will help to illustrate the potential of evolutionary algorithms in general and a quick overview of the DEAP framework's possibilities. The problem is simple and widely used in the evolutionary computational community: we will create a population of individuals consisting of integer vectors randomly filled with 0 and 1. Then we let our population evolve until one of its members contains only 1 and no 0 anymore.

## Setting Things Up

In order to solve the One Max problem, we need a bunch of ingredients. First we have to define our individuals, which will be lists of integer values, and to generate a population using them. Then we will add some functions and operators taking care of the evaluation and evolution of our population and finally put everything together in script.

But first of all, we need to import some modules.

```         
import random

!pip install deap
from deap import base
from deap import creator
from deap import tools
```

```         
Collecting deap
  Downloading deap-1.4.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (135 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m135.4/135.4 kB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from deap) (1.23.5)
Installing collected packages: deap
Successfully installed deap-1.4.1
```

## Creator

Since the actual structure of the required individuals in genetic algorithms does strongly depend on the task at hand, DEAP does not contain any explicit structure. It will rather provide a convenient method for creating containers of attributes, associated with fitnesses, called the deap.creator. Using this method we can create custom individuals in a very simple way.

The creator is a class factory that can build new classes at run-time. It will be called with first the desired name of the new class, second the base class it will inherit, and in addition any subsequent arguments you want to become attributes of your class. This allows us to build new and complex structures of any type of container from lists to n-ary trees.

```         
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
```

First we will define the class FitnessMax. It will inherit the Fitness class of the deap.base module and contain an additional attribute called weights. Please mind the value of weights to be the tuple (1.0,). This way we will be maximizing a single objective fitness. We can't repeat it enough, in DEAP single objectives is a special case of multi objectives.

Next we will create the class Individual, which will inherit the class list and contain our previously defined FitnessMax class in its fitness attribute. Note that upon creation all our defined classes will be part of the creator container and can be called directly.

## Toolbox

Now we will use our custom classes to create types representing our individuals as well as our whole population.

All the objects we will use on our way, an individual, the population, as well as all functions, operators, and arguments will be stored in a DEAP container called Toolbox. It contains two methods for adding and removing content, register() and unregister().

```         
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
```

In this code block we register a generation function toolbox.attr_bool() and two initialization ones individual() and population(). toolbox.attr_bool(), when called, will draw a random integer between 0 and 1. The two initializers, on the other hand, will instantiate an individual or population.

The registration of the tools to the toolbox only associates aliases to the already existing functions and freezes part of their arguments. This allows us to fix an arbitrary amount of argument at certain values so we only have to specify the remaining ones when calling the method. For example, the attr_bool() generator is made from the randint() function that takes two arguments a and b, with a \<= n \<= b, where n is the returned integer. Here, we fix a = 0 and b = 1.

Our individuals will be generated using the function initRepeat(). Its first argument is a container class, in our example the Individual one we defined in the previous section. This container will be filled using the method attr_bool(), provided as second argument, and will contain 100 integers, as specified using the third argument. When called, the individual() method will thus return an individual initialized with what would be returned by calling the attr_bool() method 100 times. Finally, the population() method uses the same paradigm, but we don't fix the number of individuals that it should contain.

##The Evaluation Function The evaluation function is pretty simple in our example. We just need to count the number of ones in an individual.

```         
def evalOneMax(individual):
    return sum(individual),

print(evalOneMax([0,0,0,0]))
print(evalOneMax([0,1,0,1]))
print(evalOneMax([1,1,1,1]))
```

```         
(0,)
(2,)
(4,)
```

## The Genetic Operators

Within DEAP there are two ways of using operators. We can either simply call a function from the tools module or register it with its arguments in a toolbox, as we have already seen for our initialization methods. The most convenient way, however, is to register them in the toolbox, because this allows us to easily switch between the operators if desired. The toolbox method is also used when working with the algorithms module. See the [One Max Problem: Short Version](https://deap.readthedocs.io/en/master/examples/ga_onemax_short.html#short-ga-onemax) for an example.

Registering the genetic operators required for the evolution in our One Max problem and their default arguments in the toolbox is done as follows.

```         
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
```

The evaluation will be performed by calling the alias evaluate. It is important to not fix its argument in here. We will need it later on to apply the function to each separate individual in our population. The mutation, on the other hand, needs an argument to be fixed (the independent probability of each attribute to be mutated indpb).

## Evolving the Population

Once the representation and the genetic operators are chosen, we will define an algorithm combining all the individual parts and performing the evolution of our population until the One Max problem is solved.

## Creating the Population

First of all, we need to actually instantiate our population. But this step is effortlessly done using the population() method we registered in our toolbox earlier on.

```         
pop = toolbox.population(n=300)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
```

**pop** will be a list composed of 300 individuals. Since we left the parameter **n** open during the registration of the population() method in our toolbox, we are free to create populations of arbitrary size.

Before we go on, this is the time to define some constants we will use later on.

```         
# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2
```

## Evaluating the Population

The next thing to do is to evaluate our brand new population. We map() the evaluation function to every individual and then assign their respective fitness. Note that the order in fitnesses and population is the same.

The evolution of the population is the final step we have to accomplish. Recall, our individuals consist of 100 integer numbers and we want to evolve our population until we got at least one individual consisting of only 1s and no 0s. So all we have to do is to obtain the fitness values of the individuals!

To check the performance of the evolution, we will calculate and print the minimal, maximal, and mean values of the fitnesses of all individuals in our population as well as their standard deviations.

```         
def findFitness():
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    return fits

fits=findFitness()
```

```         
  Min 34.0
  Max 64.0
  Avg 49.986666666666665
  Std 5.0139627264492495
```

## Mating and Mutation

In genetic algorithms, evolution occurs via either mutation or crossover, both of which happen (or don't happen) randomly. In mutation, we change one or more of the genes of one of our individuals. In cross-over, two individuals are mated to mix their genes.

The crossover (or mating) and mutation operators, provided within DEAP, usually take respectively 2 or 1 individual(s) as input and return 2 or 1 modified individual(s). In addition they modify those individuals within the toolbox container and we do not need to reassign their results.

We will perform both the crossover (mating) and the mutation of the produced children with a certain probability of CXPB and MUTPB. The del statement will invalidate the fitness of the modified offspring.

```         
def mateAndMutate(offspring):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

```

## The Main Loop

This will creates an offspring list, which is an exact copy of the selected individuals. The toolbox.clone() method ensure that we don't use a reference to the individuals but an completely independent instance. This is of utter importance since the genetic operators in toolbox will modify the provided objects in-place.

We then mutate and mate the individuals to find the next generation of individuals. We evaluate them, and continue until one of our individuals evolves to be the perfect organism (fitness of 100 or more), or until the number of generations reaches 1000.

At each generation, we output some statistics about that generation's population, as well as a graph of the genetic material for an arbitrary individual. A 0 in the genetic material is drawn in red, and a 1 is drawn in blue.

```         
import numpy as np
# Variable keeping track of the number of generations
g = 0

# Begin the evolution
while max(fits) < 100 and g < 1000:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    mateAndMutate(offspring)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop[:] = offspring

    # print statistics on our updated population
    fits=findFitness()

    # plot an arbitrary organism
    x = [i/100 for i in range(len(pop[0]))]
    y = [1 for i in x]
    colors = ['r' if pop[0][i]==0 else 'b' for i in range(len(pop[0]))]
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()
```

```         
-- Generation 1 --
  Min 45.0
  Max 68.0
  Avg 54.11666666666667
  Std 4.260249079833473
```

![png](Untitled0_files/Untitled0_24_1.png)

```         
-- Generation 2 --
  Min 48.0
  Max 68.0
  Avg 57.58
  Std 3.579143286691223
```

![png](Untitled0_files/Untitled0_24_3.png)

```         
-- Generation 3 --
  Min 50.0
  Max 68.0
  Avg 60.406666666666666
  Std 3.3943024156502473
```

![png](Untitled0_files/Untitled0_24_5.png)

```         
-- Generation 4 --
  Min 56.0
  Max 71.0
  Avg 62.86
  Std 2.897194044818846
```

![png](Untitled0_files/Untitled0_24_7.png)

```         
-- Generation 5 --
  Min 58.0
  Max 74.0
  Avg 64.91666666666667
  Std 2.60954444725938
```

![png](Untitled0_files/Untitled0_24_9.png)

```         
-- Generation 6 --
  Min 59.0
  Max 75.0
  Avg 66.61666666666666
  Std 2.6374967087922037
```

![png](Untitled0_files/Untitled0_24_11.png)

```         
-- Generation 7 --
  Min 57.0
  Max 78.0
  Avg 68.42333333333333
  Std 2.609748817202331
```

![png](Untitled0_files/Untitled0_24_13.png)

```         
-- Generation 8 --
  Min 62.0
  Max 78.0
  Avg 70.19666666666667
  Std 2.655432586645862
```

![png](Untitled0_files/Untitled0_24_15.png)

```         
-- Generation 9 --
  Min 63.0
  Max 79.0
  Avg 71.91
  Std 2.832295888497641
```

![png](Untitled0_files/Untitled0_24_17.png)

```         
-- Generation 10 --
  Min 66.0
  Max 80.0
  Avg 73.69666666666667
  Std 2.669205041872086
```

{{< admonition type=note title="Bibliography" open=false >}}

\## Bibliography

-   Goldberg, David E. 1989. Genetic Algorithms in Search, Optimization, and Machine Learning. New York, NY, USA: Addison-Wesley.

{{< /admonition >}}

Pics by <a href="https://www.reddit.com/user/mmmPlE/">mmmPIE in <a href="https://www.reddit.com/r/HybridAnimals/">Reddit/HybridAnimals</a>, and <a href="https://unsplash.com/es/@dulceylima">Dulcey Lima in Unsplash.