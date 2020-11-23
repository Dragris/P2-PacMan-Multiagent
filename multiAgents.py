# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import util

from numpy import mean

from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        "*** YOUR CODE HERE ***"
        # Focus on eating food. If there's a ghost ignore food
        newFood = newFood.asList()
        minFoodDist = float("inf")

        for food in newFood:
            minFoodDist = min(minFoodDist, manhattanDistance(newPos, food))

        # Avoiding ghost if too close
        for ghost in successorGameState.getGhostPositions():
            """ 
            I use manhattan distance as it's simple and effective.
            2 for ghost distance is an arbitrary number, but it's safe that in next move ghost won't catch Pacman
            """
            if manhattanDistance(newPos, ghost) < 2:
                return -float('inf')  # - infinite = yabai
        # I use the reciprocal value to important things, as hinted in task
        return successorGameState.getScore() + 1.0 / minFoodDist


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0)[0]

    def minimax(self, gameState, currentDepth, agentIndex):
        # I'm too lazy to check if all agents are done to sum depth so i multiply by agents and it's done
        # I know that the statement above is just an if, but i didn't want to think about it
        # It also helps to know which agent I am checking, and internet tutorials help you understand this other way
        # Terminal states:
        if currentDepth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Pacman time
        if agentIndex == 0:
            return self.max_value(gameState, currentDepth, agentIndex)[1]
        else:
            return self.min_value(gameState, currentDepth, agentIndex)[1]

    # Best Pacman case
    def max_value(self, gameState, currDepth, agentIndex):
        # With float -infinite I can get a better value for sure
        val = ("YAMETEKUDASTOP", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            # Recursive call for all actions, we get max
            # Max/min function with lambda to get max value and not max action name
            val = max(val, (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                                 currDepth + 1, (currDepth + 1) % gameState.getNumAgents())),
                      key=lambda x: x[1])
        return val

    # Best ghost case
    def min_value(self, gameState, currentDepth, agentIndex):
        # With float infinite I can get a better value for sure
        val = ("YAMETEKUDASTOP", float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            # Recursive call for all actions, I get min
            val = min(val, (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                                 currentDepth + 1, (currentDepth + 1) % gameState.getNumAgents())),
                      key=lambda x: x[1])
        # Literal copy-paste from above but swapping max with min
        return val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    # Same as minmax, I add pruning
    def alphabeta(self, gameState, currDepth, agentIndex, alpha, beta):
        if currDepth is self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maxval(gameState, currDepth, agentIndex, alpha, beta)[1]
        else:
            return self.minval(gameState, currDepth, agentIndex, alpha, beta)[1]

    def maxval(self, gameState, curr_depth, agentIndex, alpha, beta):
        val = ("YAMETEKUDASTOP", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            # Recursive call for all actions, I get max
            # Max/min function with lambda to get max value and not max action name
            val = max(val, (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action), curr_depth + 1,
                                                   (curr_depth + 1) % gameState.getNumAgents(), alpha, beta)),
                      key=lambda x: x[1])

            # Pruning
            if val[1] > beta:
                return val
            else:
                alpha = max(alpha, val[1])

        return val

    def minval(self, gameState, currDepth, agentIndex, alpha, beta):
        val = ("YAMETEKUDASTOP", float("inf"))  # Val siempre tendrá esta forma
        for action in gameState.getLegalActions(agentIndex):
            # Recursive call for all actions, I get min
            val = min(val, (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action), currDepth + 1,
                                                   (currDepth + 1) % gameState.getNumAgents(), alpha, beta)),
                      key=lambda x: x[1])

            # Pruning
            if val[1] < alpha:
                return val
            else:
                beta = min(beta, val[1])
        # Literal copy-paste from above but swapping max with min
        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax(gameState, 1, 0)

    def expectimax(self, gameState, currentDepth, agentIndex):
        # Check for terminal states
        if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # I get all legal actions except stop action, I don't like that one, it's always there
        legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action != 'Stop']

        nextIndex, nextDepth = agentIndex + 1, currentDepth
        # When I get back to check Pac-man I add one to depth and start checking again
        if nextIndex >= gameState.getNumAgents():
            nextIndex = 0
            nextDepth += 1
        # I check every legal move, basically the recursive part of the algorithm
        results = [self.expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex) for action
                   in legalMoves]

        # Here we choose between the different probability nodes in each search
        if agentIndex == 0 and currentDepth == 1:
            bestMove = max(results)
            bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
            # Here I pick the random node
            chosenIndex = random.choice(bestIndices)
            return legalMoves[chosenIndex]

        elif agentIndex == 0:
            # With Pac-man I get the best move, I want to win
            bestMove = max(results)
            return bestMove
        else:
            # With ghosts I don't get the best or worst move, it's the average
            bestMove = mean(results)
            return bestMove


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    I'm evaluating by:
        Close food
        Food left
        Capsules left
        Distance to closest ghost
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    # I check for food left and capsules left
    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    # Checking closest food
    minFoodDist = float('inf')
    for food in newFood:
        minFoodDist = min(minFoodDist, manhattanDistance(newPos, food))

    # Avoiding ghost if too close, I like to stay alive
    minGhostDist = 0
    for ghost in currentGameState.getGhostPositions():
        """ 
        We use manhattan distance as it's simple and effective.
        2 is arbitrary number, but it's safe that in next move ghost won't catch Pacman
        """
        minGhostDist = manhattanDistance(newPos, ghost)
        if minGhostDist < 2:
            return -float('inf')  # - infinite = yabai

    """
    We add some multipliers to manually adjust the behaviour of pac-man
    
    With these parameters we want Pac-man to first gobble when there's a lot of food.
    If there are capsules we want to use them. And then if there's food close we should eat it.
    We won't be hunting ghosts to maximize score as it's not really needed, but could be done
    with the scare times.
    
    Also we borrowed from the reflex agent the keeping alive thing so we don't die like idiots.
    """
    foodLeftMult = 50000
    capsLeftMult = 5000
    minFoodDistMult = 250

    # I add 1 to every Left so I don't divide by 0
    # If minGhostDist <= 2 then I'm always escaping, there's no needed multiplier
    return 1.0 / (foodLeft + 1) * foodLeftMult + minGhostDist + \
           1.0 / (minFoodDist + 1) * minFoodDistMult + \
           1.0 / (capsLeft + 1) * capsLeftMult


# Abbreviation
better = betterEvaluationFunction
