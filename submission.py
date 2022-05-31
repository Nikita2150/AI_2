from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random

import time

class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        curr_taxi = env.get_taxi(taxi_id)
        def fNeed():
            fuel = curr_taxi.fuel
            if env.taxi_is_occupied(taxi_id):
                dest = curr_taxi.passenger.destination
                md = manhattan_distance(dest, curr_taxi.position)
            else:
                md = 16
                passenger = None
                for p in env.passengers:
                    temp_md = manhattan_distance(p.position, curr_taxi.position)
                    if temp_md < md:
                        md = temp_md
                        passenger = p
                md += manhattan_distance(passenger.position, passenger.destination)
            
            if md > fuel:
                #find closest gas station
                md_g = 16
                for g in env.gas_stations:
                    temp_md_g = manhattan_distance(g.position, curr_taxi.position)
                    if temp_md_g < md_g:
                        md_g = temp_md_g
                return 8 - md_g
            return 0
            
        def diffCash():
            taxi = env.get_taxi(taxi_id)
            other_taxi = env.get_taxi((taxi_id+1) % 2)
            temp_ret =  taxi.cash - other_taxi.cash
            if temp_ret > 0:
                return temp_ret
            return 0
            
        def isPass():
            if env.taxi_is_occupied(taxi_id):
                return 30
            return 0
        def canDrop():
            curr_taxi = env.get_taxi(taxi_id)
            if isPass() > 0 and manhattan_distance(curr_taxi.position, curr_taxi.passenger.destination) == 0:
                return 3
            return 0
            
        def dist():
            curr_taxi = env.get_taxi(taxi_id)
            if env.taxi_is_occupied(taxi_id):
                dest = curr_taxi.passenger.destination
                md = manhattan_distance(dest, curr_taxi.position)
            else:
                md = 16
                
                for p in env.passengers:
                    temp_md = manhattan_distance(p.position, curr_taxi.position)
                    if temp_md < md:
                        md = temp_md
            return 8 - md
         
        def refuel():
            if fNeed() == 0:
                return 100
            return 0
            
        #print("f: ", str(10*fNeed()), "diff:", str(30 * diffCash()), "pass:", str(isPass()), "drop:", str(canDrop()), "dist:", str(dist()), "ref:", str(refuel()), "tot:",str(10*fNeed() + 30 * diffCash() + isPass() + canDrop() + dist() +refuel()))
        return 10*fNeed() + 30 * diffCash() + isPass() + canDrop() + dist() +refuel()
  
class TimeStopped(Exception):
    pass
class AgentMinimax(Agent):
    # TODO: section b : 1
    
    def heuristic(self, env: TaxiEnv, taxi_id: int):
        curr_taxi = env.get_taxi(taxi_id)
        def fNeed():
            fuel = curr_taxi.fuel
            if env.taxi_is_occupied(taxi_id):
                dest = curr_taxi.passenger.destination
                md = manhattan_distance(dest, curr_taxi.position)
            else:
                md = 16
                passenger = None
                for p in env.passengers:
                    temp_md = manhattan_distance(p.position, curr_taxi.position)
                    if temp_md < md:
                        md = temp_md
                        passenger = p
                md += manhattan_distance(passenger.position, passenger.destination)
            
            if md > fuel:
                #find closest gas station
                md_g = 16
                for g in env.gas_stations:
                    temp_md_g = manhattan_distance(g.position, curr_taxi.position)
                    if temp_md_g < md_g:
                        md_g = temp_md_g
                return 8 - md_g
            return 0
            
        def diffCash():
            taxi = env.get_taxi(taxi_id)
            other_taxi = env.get_taxi((taxi_id+1) % 2)
            temp_ret =  taxi.cash - other_taxi.cash
            if temp_ret > 0:
                return temp_ret
            return 0
            
        def isPass():
            if env.taxi_is_occupied(taxi_id):
                return 30
            return 0
        def canDrop():
            curr_taxi = env.get_taxi(taxi_id)
            if isPass() > 0 and manhattan_distance(curr_taxi.position, curr_taxi.passenger.destination) == 0:
                return 3
            return 0
            
        def dist():
            curr_taxi = env.get_taxi(taxi_id)
            if env.taxi_is_occupied(taxi_id):
                dest = curr_taxi.passenger.destination
                md = manhattan_distance(dest, curr_taxi.position)
            else:
                md = 16
                
                for p in env.passengers:
                    temp_md = manhattan_distance(p.position, curr_taxi.position)
                    if temp_md < md:
                        md = temp_md
            return 8 - md
         
        def refuel():
            if fNeed() == 0:
                return 100
            return 0
            
        #print("f: ", str(10*fNeed()), "diff:", str(30 * diffCash()), "pass:", str(isPass()), "drop:", str(canDrop()), "dist:", str(dist()), "ref:", str(refuel()), "tot:",str(10*fNeed() + 30 * diffCash() + isPass() + canDrop() + dist() +refuel()))
        return 10*fNeed() + 30 * diffCash() + isPass() + canDrop() + dist() +refuel()
     
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        time_limit = time_limit*0.7
        start = time.time()
        depth = 1
        op = None   
        if env.num_steps == 1:
            #for every time_limit it should just work.. if we'll get TimeStopped exception, it's from here
            _, op = self.value(env.clone(), taxi_id, 1, time_limit, start, env.num_steps, 1)
            return op
        while time.time() - start < time_limit and depth <= env.num_steps:
            #print(depth, env.num_steps)
            new_env = env.clone()
            new_env.num_steps = depth
            try:
               temp = self.value(new_env, taxi_id, 1, time_limit, start, env.num_steps, depth)
               _, op_t = temp 
               if op_t == None:
                return op
               op = op_t
               
               # print(flag, op)
               if depth == 1:
                    depth = 0 # it does weird things when min is the last tree
               depth += 2
            except TimeStopped:
                return op
        return op
        
    def max_value(self, env: TaxiEnv, taxi_id, time_limit, start, num_steps, depth):
        if time.time() - start > time_limit:
            #we should stop here.. and give the prev answer as the answer
            raise TimeStopped
        last_op = None
        #init successors
        
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
 
        v, _ = self.value(children[0].clone(), 1-taxi_id, 0, time_limit, start, num_steps, depth)
        v -= 1
        
 
        for child, op in zip(children, operators):
            temp_v, _ = self.value(child, 1-taxi_id, 0, time_limit, start, num_steps, depth)
            if temp_v > v:
                v = temp_v
                last_op = op
        return v, last_op
        
        
    def min_value(self, env: TaxiEnv, taxi_id, time_limit, start, num_steps, depth):
        if time.time() - start > time_limit:
            #we should stop here.. and give the prev answer as the answer
            raise TimeStopped
        last_op = None
        
        #init successors
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        
        v, _ = self.value(children[0].clone(), 1-taxi_id, 1, time_limit, start, num_steps, depth) #initalizing
        v += 1 # val + 1 - so we could find at least one child in the loop
        for child, op in zip(children, operators):
            temp = self.value(child, 1-taxi_id, 1, time_limit, start, num_steps, depth)
            temp_v, _ = temp
            if temp_v < v:
                v = temp_v
                last_op = op
            
        return v, last_op
    
    def value(self, env: TaxiEnv, taxi_id, min_or_max, time_limit, start, num_steps, depth):
        if time.time() - start > time_limit:
            #we should stop here.. and give the prev answer as the answer
            raise TimeStopped
        env = env.clone()
        
        
        #validate terminal state
        if len([taxi for taxi in env.taxis if taxi.fuel > 0]) == 0 or (env.num_steps <= 0 and num_steps <= depth):

            taxi = env.get_taxi(taxi_id)
            other_taxi = env.get_taxi((taxi_id+1) % 2)
            temp_ret =  taxi.cash - other_taxi.cash
            if temp_ret < 0:
                return 0, None
            return temp_ret, None
        elif env.done(): #we're done because the depth is big enough, but have to use heuristic
            return self.heuristic(env, taxi_id), None
        if min_or_max == 0: #min
            return self.min_value(env, taxi_id, time_limit, start, num_steps, depth)
        return self.max_value(env, taxi_id, time_limit, start, num_steps, depth)


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
