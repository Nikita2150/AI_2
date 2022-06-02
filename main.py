import random
import time

from TaxiEnv import TaxiEnv
import argparse
import submission
import Agent


def run_agents():
    
    agents = {
        "random": Agent.AgentRandom(),
        "greedy": Agent.AgentGreedy(),
        "greedy_improved": submission.AgentGreedyImproved(),
        "minimax": submission.AgentMinimax(),
        "alphabeta": submission.AgentAlphaBeta(),
        "expectimax": submission.AgentExpectimax()
    }
    
    max_b = [0, 0, 0, 0]
    max_seed_a = [0, 0, 0, 0]
    sum_win = [0, 0, 0, 0]
    sum_draw = [0, 0, 0, 0]
    sum_lose = [0, 0, 0, 0]
    sum_score = [0, 0, 0, 0]
    times = [0, 0, 0, 0]
    """ We'll test all of our new agents against the greedy one """
    bot = "greedy"
    agent_names_l = [ ["greedy_improved", bot], ["minimax", bot], ["alphabeta", bot], ["expectimax", bot]]
    
    min_seed = 0
    max_seed = 200

    time_limit = 0.1
    num_steps = 20
    
    for i, agent_names in enumerate(agent_names_l):
        print("Running",agent_names[0],"against",agent_names[1]+": ",end="",flush=True)
        global_str_time = time.time()
        for s in range(min_seed, max_seed + 1):
            if s%round((max_seed+min_seed)/5) == 0:
                print(str(s)+" - "+str(min(s+round((max_seed+min_seed)/5)-1, max_seed)), end="",flush=True)
                if s == max_seed:
                    print()
                else:
                    print(", ",end="",flush=True)
                
            env = TaxiEnv()
            env.generate(s, 2*num_steps)
    
    
            for _ in range(num_steps):
                for ai, agent_name in enumerate(agent_names):
                    agent = agents[agent_name]
                    start = time.time()
                    op = agent.run_step(env, ai, time_limit)
                    end = time.time()
                    if end - start > time_limit:
                        raise RuntimeError("Agent used too much time!")
                    env.apply_operator(ai, op)
                    
                if env.done():
                    break
            balances = env.get_balances()
            if balances[0] > max_b[i]:
                max_b[i] = balances[0]
                max_seed_a[i] = s
            sum_score[i] += balances[0]
            if balances[0] == balances[1]:
                sum_draw[i] += 1
            elif balances[0] > balances[1]:
                sum_win[i] += 1
            else:
                sum_lose[i] += 1
                
        times[i] = time.time() - global_str_time
        
        print("Results for",agent_names[0]+":\nWins:", sum_win[i], "Loses:",sum_lose[i],"Draws:",sum_draw[i],"Total Score:",sum_score[i], "Maximal balance",str(max_b[i])+", in seed",max_seed_a[i],end="")
        print(". Elapsed time: {0:.2f}s, WIN RATE: {1:.2f}%".format(times[i], (sum_win[i]/(sum_win[i]+sum_lose[i]+sum_draw[i]))*100))
        print()
if __name__ == "__main__":
    run_agents()
