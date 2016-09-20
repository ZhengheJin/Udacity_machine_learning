import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.reward = 0
        self.next_waypoint = None
				
        self.moves = 0
        self.qDict = dict()
        self.alpha = 0.9
        self.epsilon = 0.95
        self.gamma = 0.2

        self.state = None
        self.new_state = None
            
        self.cum_reward = 0
        self.possible_actions = Environment.valid_actions
        self.action = None


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None

        self.moves = 0
        self.new_state = None
        self.cum_reward = 0
        self.action = None
	
	def getQvalue(self, state, action):
		return self.qDict.get((state, action), 5.0)

	def getMaxQ(self, state):
		q = [self.getQvalue(stata, a) for a in self.possible_actions]
		return max(q)
	
	def get_action(self, state):
		if random.random() < self.epsilon:
			action = random.choice(self.possible_actions)
		else:
			q = [self.getQvalue(state, a) for a in self.possible_actions]
			if q.count(max(q)) > 1: 
				best_actions = [i for i in range(len(self.possible_actions)) if q[i] == max(q)]                       
				index = random.choice(best_actions)

			else:
				index = q.index(max(q))
			action = self.possible_actions[index]

		return action

	def qlearning(self, state, action, nextState, reward):
		"""
		use Qlearning algorithm to update q values
		"""
		key = (state, action)
		if (key not in self.qDict):
			# initialize the q values
			self.qDict[key] = 5.0
		else:
			self.qDict[key] = self.qDict[key] + self.alpha * (reward + self.gamma*self.getMaxQ(nextState) - self.qDict[key])

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        is_action = True
        if self.next_waypoint == 'right' and inputs['light'] == 'red' and inputs['left'] == 'forward':
            is_action = False
        elif self.next_waypoint == 'straight' and inputs['light'] == 'red':
            is_action = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right' :
                is_action = False


        # Execute action and get reward
        self.new_state = inputs
        self.new_state['next_waypoint'] = self.next_waypoint
        self.new_state = tuple(sorted(self.new_state.items()))


        # TODO: Select action according to your policy
        action = None
        if is_action:
			if random.random() < self.epsilon:
				action = random.choice(self.possible_actions)
			else:
				q = [self.getQvalue(state, a) for a in self.possible_actions]
				if q.count(max(q)) > 1: 
					best_actions = [i for i in range(len(self.possible_actions)) if q[i] == max(q)]                       
					index = random.choice(best_actions)
				else:
					index = q.index(max(q))
				action = self.possible_actions[index]
			
			new_reward = self.env.act(self, action)
			if self.reward != None:
				key = (self.state, self.action)
				if (key not in self.qDict):
					# initialize the q values
					self.qDict[key] = 5.0
				else:
					self.qDict[key] = self.qDict[key] + self.alpha * (self.reward + self.gamma*self.getMaxQ(self.nextState) - self.qDict[key])
				#self.qlearning(self.state, self.action, self.new_state, self.reward)
			self.action = action
			self.state = self.new_state
			self.reward = new_reward
			self.cum_reward += self.reward
			self.moves += 1

        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
