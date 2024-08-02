# valueIterationAgents.py
# -----------------------
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
import queue

# valueIterationAgents.py
# -----------------------
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


import mdp, util
from queue import PriorityQueue

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def getGreedyUpdate(self, state):
        """computes a one step-ahead value update and return it"""
        if self.mdp.isTerminal(state):
            return self.values[state]
        actions = self.mdp.getPossibleActions(state)
        vals = util.Counter()
        for action in actions:
            vals[action] = self.computeQValueFromValues(state, action)
        return max(vals.values())

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            valuesK = self.values.copy()
        
            for state in self.mdp.getStates():
                qValue = util.Counter() 
                for action in self.mdp.getPossibleActions(state):
                    qValue[action] = self.computeQValueFromValues(state, action)
                valuesK[state] = qValue[qValue.argMax()]

            self.values = valuesK

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        probPairs = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0
        for nextState, prob in probPairs:
            reward = self.mdp.getReward(state, action, nextState)
            total += prob * (reward + self.discount * self.values[nextState])
        return total


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        maxVal = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maxVal:
                maxVal = qValue
                bestAction = action
        return bestAction



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()

        pq = util.PriorityQueue()
        
        for state in self.mdp.getStates():
            qValue = util.Counter()

            for action in self.mdp.getPossibleActions(state):
                trans = self.mdp.getTransitionStatesAndProbs(state, action)
                for (nextState, prob) in trans:
                    if prob != 0:
                        predecessors[nextState].add(state)
                qValue[action] = self.computeQValueFromValues(state, action)

            if not self.mdp.isTerminal(state): 
                maxQValue = qValue[qValue.argMax()]
                diff = abs(self.values[state] - maxQValue)
                pq.update(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                return

            state = pq.pop()

            if not self.mdp.isTerminal(state):
                qValue = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    qValue[action] = self.computeQValueFromValues(state, action)
                self.values[state] = qValue[qValue.argMax()]

            for p in predecessors[state]:
                qPreValues = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                    qPreValues[action] = self.computeQValueFromValues(p, action)

                maxQP = qPreValues[qPreValues.argMax()]
                diff = abs(self.values[p] - maxQP)
                
                if diff > self.theta:
                    pq.update(p, -diff)