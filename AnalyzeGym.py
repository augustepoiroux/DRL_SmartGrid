import numpy as np
import matplotlib.pyplot as plt
import copy

from EnvGym import Env, ACTIONS, BATTERY_CAPACITY

"""
Strategies supported :
    - RL agent
    - Nothing
    - Random
    - Trade
    - RandomBattery    # random charge/discharge
    - SmartBattery
    - SmartBattery2
"""
STRATEGIES = [
    "Random",
    "Nothing",
    "Trade",
    "RandomBattery",
    "SmartBattery",
    "SmartBattery2",
    "Model",
]

NB_ACTION = len(ACTIONS)


def strategyAction(strategy, state, model=None):
    """ 
    Determines the action of the given strategy on the state.

    Parameters: 
    strategy (str): strategy to use

    state (State): current State of the environment

    Returns: 
    action (str): action of the strategy given the state.

    """
    if strategy == "Model":
        if model is None:
            print("No model given in the function strategyAction")
            return ACTIONS.index("nothing")

        return model.predict(state.toArray(), deterministic=True)[0]

    if strategy == "Random":
        return np.random.randint(NB_ACTION)

    if strategy == "Trade":
        return ACTIONS.index("trade")

    if strategy == "Nothing":
        return ACTIONS.index("nothing")

    if strategy == "RandomBattery":
        return np.random.choice([ACTIONS.index("charge"), ACTIONS.index("discharge")])

    if strategy == "SmartBattery":
        if state.panelProd > state.consumption:
            return ACTIONS.index("charge")
        else:
            return ACTIONS.index("discharge")

    if strategy == "SmartBattery2":
        if state.panelProd > state.consumption and state.battery < BATTERY_CAPACITY * 0.9999:
            return ACTIONS.index("charge")
        elif state.panelProd > state.consumption:
            return ACTIONS.index("trade")
        else:
            return ACTIONS.index("discharge")


def test(env: Env, model=None):
    """ 
    Displays figures to compare the result of the DQN algorithm to other predefined strategies.

    Parameters: 
    env (Env): environment on which the strategies are tested

    model: the model returned by the DQN algorithm.
    If this parameter is set to None, then only the predefined strategies are tested.
    This is useful to check the environment.

    """
    env.reset()
    initState = copy.deepcopy(env.currentState)

    conso, prod, price = [], [], []

    for i in range(env.epLength):
        env.step(0)

        conso.append(env.currentState.consumption)
        prod.append(env.currentState.panelProd)
        price.append(env.currentState.price)

    actions, cost = {}, {}
    battery, charge, discharge, generate, trade = {}, {}, {}, {}, {}

    strategies_list = STRATEGIES[:]

    if model is None:
        strategies_list.remove("Model")

    for strategy in strategies_list:
        actions[strategy], cost[strategy] = [], []
        (battery[strategy], charge[strategy], discharge[strategy], trade[strategy],) = (
            [],
            [],
            [],
            [],
        )

        env.currentState = copy.deepcopy(initState)
        for i in range(env.epLength):
            action = strategyAction(strategy, env.currentState, model)
            _, _, _, info = env.step(action)
            step_cost = info["cost"]

            cost[strategy].append(step_cost)
            actions[strategy].append(action)
            battery[strategy].append(env.currentState.battery)

            charge[strategy].append(env.currentState.charge)
            discharge[strategy].append(env.currentState.discharge)
            trade[strategy].append(env.currentState.trade)

    for strategy in strategies_list:
        print(f"{strategy:<20} cost: {np.sum(cost[strategy])}")

    ## Issues with Qt (xcb)
    # fig, axs = plt.subplots(len(strategies_list))
    # for i, s in enumerate(strategies_list):
    #     axs[i].plot(trade[s])
    #     axs[i].plot(battery[s])
    #     axs[i].legend(["Trade", "Battery"])
    #     axs[i].title.set_text(s)
    # plt.figure(1)

    # fig, axs = plt.subplots(len(strategies_list))

    # for i, s in enumerate(strategies_list):
    #     axs[i].plot(actions[s])
    #     axs[i].legend(["Actions"])
    #     axs[i].title.set_text(s)
    # plt.figure(2)

    # fig, axs = plt.subplots(2)
    # axs[0].plot(conso)
    # axs[0].plot(prod)
    # axs[1].plot(price)
    # axs[0].legend(["Consumption", "Production"])
    # axs[1].title.set_text("Price")
    # plt.figure(3)

    # fig, ax = plt.subplots()
    # for s in strategies_list:
    #     ax.plot(np.cumsum(cost[s]))

    # ax.legend(strategies_list)
    # ax.title.set_text("Cost")
    # plt.figure(4)

    # plt.show()
