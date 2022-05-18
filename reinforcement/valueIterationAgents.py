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
 
    def runValueIteration(self):
        #Note: I think we should use the util.Counter thing?
        #for times in range(0, iterations): #values from previous iteration so we don't update over them while iterating
            #prevVals = self.values.copy() #iterate through all states
            #for state in mdp.getStates(): #will store the action-value for the iteration
                #value = util.Counter()
                #for action in mdp.getPossibleActions(state):
                    #for transitionState, probability in mdp.getTransitionStatesAndProbs(state, action):
                    #expected value, probability * reward for the state with the discount * reward
                        #value[action] += probability * (mdp.getReward( state, action, transitionState) + discount * prevVals[transitionState])
                        #update the values to the new value from the iteration
                #self.values[state] = value[value.argMax()]#the .argMax() function returns the one with the largest value
 
        for _ in range(self.iterations):
          newValues = self.values.copy()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              continue
            actions = self.mdp.getPossibleActions(state)
            bestValue = max([self.getQValue(state, a) for a in actions])
            newValues[state] = bestValue
          self.values = newValues
 
 
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
        #util.raiseNotDefined()
 
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qvalue = 0
        for nextstate, prob in statesAndProbs:
          reward = self.mdp.getReward(state, action, nextstate)
          q = prob * (reward + self.discount * self.values[nextstate])
          qvalue += q
        return qvalue
 
 
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
 
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
 
        if self.mdp.isTerminal(state):
          return None
        actions = self.mdp.getPossibleActions(state)
        policy = util.Counter()
        for action in actions:
          policy[action] = self.getQValue(state, action)
        return policy.argMax()
 
 
    def getPolicy(self, state):
        return self.computeActionFromValues(state)
 
    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)
 
    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
 
class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
 
        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.
 
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
 
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
 
        states = self.mdp.getStates()
        nums_states = len(states)
        for i in range(self.iterations):
          state = states[i % nums_states]
          if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state)
            bestValue = max([self.getQValue(state, a) for a in actions])
            self.values[state] = bestValue
 
 
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
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
 
        states = self.mdp.getStates()
        predecessors = {}
        for state in states:
          curr_pred = set()
          for st in states:
            for action in self.mdp.getPossibleActions(st):
              for next, prob in self.mdp.getTransitionStatesAndProbs(st, action):
                if (next == state and prob != 0):
                  curr_pred.add(st)
          predecessors[state] = curr_pred
 
 
        #Initialize an empty priority queue
        priority = util.PriorityQueue()
 
        #Do stuff for each nonterminal state s
        for state in states:
          if not self.mdp.isTerminal(state):
            curr_val = self.values[state]
            q_val = []
            for action in self.mdp.getPossibleActions(state):
              q_val += [self.computeQValueFromValues(state, action)]
            max_q_val = max(q_val)
            diff = abs((curr_val - max_q_val))
            priority.push(state, -diff)
 
        #Do stuff for self.iteration
        for i in range(0, self.iterations):
          if priority.isEmpty():
            break
          s = priority.pop()
          if not self.mdp.isTerminal(s):
            vals = []
            for action in self.mdp.getPossibleActions(s):
              val = 0
              for next, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                val += prob * (self.mdp.getReward(s, action, next) + (self.discount * self.values[next]))
              vals.append(val)
            self.values[s] = max(vals)
 
          #Do stuff for each predecesser of s
          for pred in predecessors[s]:
            curr_val = self.values[pred]
            q_val = []
            for action in self.mdp.getPossibleActions(pred):
              q_val += [self.computeQValueFromValues(pred, action)]
            max_q_val = max(q_val)
            diff = abs((curr_val - max_q_val))
            if (diff > self.theta):
              priority.update(pred, -diff)
