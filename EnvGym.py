import pandas
import numpy as np
import datetime
import gym
from gym import spaces

ACTIONS = ["charge", "discharge", "nothing", "trade"]
NB_STEPS_MEMORY = 50
BATTERY_CAPACITY = 10.0

# Test
T_c = 5
T_p = 10


class State:
    def __init__(self):
        self.battery = 0.0
        self.panelProd = 0.0
        self.consumption = 0.0
        self.price = 0.0
        self.row = 0

        self.charge = 0.0
        self.discharge = 0.0
        self.trade = 0.0

        self.panelProdMemory = [0.0] * NB_STEPS_MEMORY
        self.consumptionMemory = [0.0] * NB_STEPS_MEMORY
        self.priceMemory = [0.0] * NB_STEPS_MEMORY

    def updateMemory(self):
        """ 
        The state memorize values of production, consumption and price over the last NB_STEPS_MEMORY steps.
        This function has to be called each time these parameters are updated.
        """
        self.panelProdMemory.pop(0)
        self.panelProdMemory.append(self.panelProd)
        self.consumptionMemory.pop(0)
        self.consumptionMemory.append(self.consumption)
        self.priceMemory.pop(0)
        self.priceMemory.append(self.price)

    def toArray(self):
        """ 
        Builds a np.array describing the essential values of the current state of the environment.
        The array generated in this function is expected to be used by the DQN algorithm.
    
        Returns: 
        np.array:  state of the environment
    
        """
        return np.array(
            [self.battery] + self.panelProdMemory + self.consumptionMemory + self.priceMemory
        )


DIM_STATE = len(State().toArray())


class Env(gym.Env):
    def __init__(self, dataFile: str, epLength=300):
        """
        Constants of the environment are defined here.
        Preprocessing of the data from dataFile.
    
        Parameters: 
        dataFile (str): a CSV file containing values of production, consumption and price over time 
    
        """
        super(Env, self).__init__()

        self.epLength = epLength

        # load data (csv)
        df = pandas.read_csv(dataFile, sep=",", header=0)

        self.data = df.values

        self.panelProdMax = max(self.data[:, 5]) / 1.5
        self.consumptionMax = max(self.data[:, 4])
        self.priceMax = max(abs(self.data[:, 3]))

        self.data[:, 5] /= self.panelProdMax
        self.data[:, 4] /= self.consumptionMax
        self.data[:, 3] /= self.priceMax

        self.batteryCapacity = BATTERY_CAPACITY
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(DIM_STATE,), dtype=np.float32,
        )

        self.reset()

    def reset(self):
        """ 
        Resets randomly the current state of the environment.
    
        Parameters: 
        nb_step (int): maximum number of steps expected after the call to this function.
        This parameter is necessary since there are no terminal states and the data is not generated.
    
        """

        self.countStep = 0

        self.currentState = State()
        row = np.random.randint(NB_STEPS_MEMORY, len(self.data) - self.epLength - 1)
        for self.currentState.row in range(row - NB_STEPS_MEMORY, row + 1):
            self.currentState.price = self.data[self.currentState.row, 3]
            self.currentState.consumption = self.data[self.currentState.row, 4]
            self.currentState.panelProd = self.data[self.currentState.row, 5]
            self.currentState.updateMemory()

        return self.currentState.toArray()

    def step(self, action):
        """ 
        Does the given action, and updates the environment accordingly.
    
        Parameters: 
        action (str): the action to do

        Returns: 
        reward (float):  reward associated to the current state and action
        
        state_updated (State): the new state of the environment

        trade_cost (float): cost/earning due to the trading

        """

        self.diffProd = self.currentState.panelProd - self.currentState.consumption
        reward = 0.0
        cost = 0.0
        self.currentState.charge = 0.0
        self.currentState.discharge = 0.0
        self.currentState.trade = 0.0

        if ACTIONS[action] == "charge":
            if self.diffProd > 0:
                self.currentState.charge = min(
                    self.diffProd, (self.batteryCapacity - self.currentState.battery),
                )
                self.currentState.battery += self.currentState.charge
                self.diffProd -= self.currentState.charge
                reward += self.currentState.charge * self.currentState.price

        elif ACTIONS[action] == "discharge":
            if self.diffProd < 0:
                self.currentState.discharge = max(self.diffProd, -self.currentState.battery)
                self.currentState.battery += self.currentState.discharge
                self.diffProd -= self.currentState.discharge

        elif ACTIONS[action] == "nothing":
            if self.diffProd < 0:
                reward -= 10.0

        self.currentState.trade = -self.diffProd

        if self.diffProd < 0:
            cost -= self.diffProd * self.currentState.price
        else:
            cost -= self.diffProd * self.currentState.price / 10

        reward -= cost

        row = self.currentState.row + 1

        self.currentState.row = row
        # self.currentState.price = 1.0
        self.currentState.price = self.data[row, 3]

        self.currentState.consumption = self.data[row, 4]
        self.currentState.panelProd = self.data[row, 5]

        # Test
        # self.currentState.consumption = np.cos(2 * np.pi * self.currentState.row / T_c)
        # self.currentState.panelProd = np.cos(2 * np.pi * self.currentState.row / T_p)

        self.currentState.updateMemory()

        self.countStep += 1
        done = self.countStep == self.epLength

        return self.currentState.toArray(), reward, done, {"cost": cost}
