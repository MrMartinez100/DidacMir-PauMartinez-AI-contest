# baseline_team.py
# ---------------
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

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class ReflexCaptureAgent(CaptureAgent):

    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        if my_state.is_pacman and Directions.STOP in actions and len(actions) > 1:
            actions = [a for a in actions if a != Directions.STOP]
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):

    RETURN_FOOD_THRESHOLD = 1

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:
            min_dist_food = min(self.get_maze_distance(my_pos, f) for f in food_list)
            features['distance_to_food'] = min_dist_food
        else:
            features['distance_to_food'] = 0
            features['no_enemy_food'] = 1
        carrying = my_state.num_carrying
        features['carrying'] = carrying
        walls = successor.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        frontier_x = mid_x - 1 if self.red else mid_x
        frontier_positions = [(frontier_x, y) for y in range(height) if not walls[frontier_x][y]]
        if frontier_positions:
            frontier_dist = min(self.get_maze_distance(my_pos, f) for f in frontier_positions)
            features['distance_to_frontier'] = frontier_dist
        else:
            features['distance_to_frontier'] = 0
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if (not e.is_pacman) and e.get_position() is not None]
        if len(ghosts) > 0:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            features['closest_ghost'] = min(ghost_dists)
        else:
            features['closest_ghost'] = 10
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def get_weights(self, game_state, action):
        carrying = game_state.get_agent_state(self.index).num_carrying
        score = self.get_score(game_state)
        t = float(max(1, self.RETURN_FOOD_THRESHOLD))
        frac = min(1.0, carrying / t)
        w_food_eat = -3.0
        w_front_eat = -0.2
        w_food_ret = -0.5
        w_front_ret = -2.5
        if score < 0:
            w_food_eat = -3.5
            w_food_ret = -0.7
            w_front_ret = -2.0
        w_food = (1 - frac) * w_food_eat + frac * w_food_ret
        w_frontier = (1 - frac) * w_front_eat + frac * w_front_ret
        return {
            'successor_score': 50.0,
            'distance_to_food': w_food,
            'carrying': 2.0 + 3.0 * frac,
            'distance_to_frontier': w_frontier,
            'closest_ghost': 2.0,
            'no_enemy_food': 10.0,
            'stop': -10,
            'reverse': -3
        }

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        if carrying > 0:
            near_border = abs(my_pos[0] - mid_x) <= 1
            if near_border:
                crossing_actions = []
                for a in actions:
                    if a == Directions.STOP:
                        continue
                    succ = game_state.generate_successor(self.index, a)
                    succ_pos = succ.get_agent_state(self.index).get_position()
                    if self.red:
                        if succ_pos[0] <= mid_x - 1:
                            crossing_actions.append(a)
                    else:
                        if succ_pos[0] >= mid_x:
                            crossing_actions.append(a)
                if crossing_actions:
                    return random.choice(crossing_actions)
        in_our_side = (self.red and my_pos[0] <= mid_x - 1) or ((not self.red) and my_pos[0] >= mid_x)
        if carrying > 0 and in_our_side:
            target = self.start
            best_dist = float('inf')
            best_actions = []
            for a in actions:
                if a == Directions.STOP:
                    continue
                succ = game_state.generate_successor(self.index, a)
                succ_pos = succ.get_agent_state(self.index).get_position()
                dist = self.get_maze_distance(succ_pos, target)
                if dist < best_dist:
                    best_dist = dist
                    best_actions = [a]
                elif dist == best_dist:
                    best_actions.append(a)
            if best_actions:
                return random.choice(best_actions)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        if Directions.STOP in best_actions and len(best_actions) > 1:
            best_actions = [a for a in best_actions if a != Directions.STOP]
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                succ = self.get_successor(game_state, action)
                pos2 = succ.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        return random.choice(best_actions)


class DefensiveReflexAgent(ReflexCaptureAgent):

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        enemies_idx = self.get_opponents(game_state)
        enemies = [game_state.get_agent_state(i) for i in enemies_idx]
        safe_actions = []
        for a in actions:
            succ = game_state.generate_successor(self.index, a)
            succ_pos = succ.get_agent_state(self.index).get_position()
            sx, sy = int(succ_pos[0]), int(succ_pos[1])
            if self.red:
                if sx <= mid_x - 1:
                    safe_actions.append(a)
            else:
                if sx >= mid_x:
                    safe_actions.append(a)
        if not safe_actions:
            safe_actions = [a for a in actions if a != Directions.STOP] or actions
        actions = safe_actions
        if self.red:
            invaders = [
                e for e in enemies
                if e.is_pacman and e.get_position() is not None and e.get_position()[0] <= mid_x - 1
            ]
            enemies_on_our_side = [
                e for e in enemies
                if e.get_position() is not None and e.get_position()[0] <= mid_x - 1
            ]
        else:
            invaders = [
                e for e in enemies
                if e.is_pacman and e.get_position() is not None and e.get_position()[0] >= mid_x
            ]
            enemies_on_our_side = [
                e for e in enemies
                if e.get_position() is not None and e.get_position()[0] >= mid_x
            ]
        if len(invaders) > 0:
            target = min(invaders, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            tx, ty = target.get_position()
            target_pos = (int(tx), int(ty))
            best_dist = float('inf')
            best_actions = []
            for a in actions:
                if a == Directions.STOP:
                    continue
                succ = game_state.generate_successor(self.index, a)
                succ_pos = succ.get_agent_state(self.index).get_position()
                sx, sy = int(succ_pos[0]), int(succ_pos[1])
                if self.red and sx > mid_x - 1:
                    continue
                if (not self.red) and sx < mid_x:
                    continue
                d = self.get_maze_distance((sx, sy), target_pos)
                if d < best_dist:
                    best_dist = d
                    best_actions = [a]
                elif d == best_dist:
                    best_actions.append(a)
            if best_actions:
                return random.choice(best_actions)
        frontier_x = int(mid_x - 1 if self.red else mid_x)
        fx, fy = int(my_pos[0]), int(my_pos[1])
        at_frontier = (fx == frontier_x)
        frontier_cells = [(frontier_x, y) for y in range(height) if not walls[frontier_x][y]]
        if len(enemies_on_our_side) > 0:
            visible_enemies = enemies_on_our_side
        else:
            visible_enemies = [e for e in enemies if e.get_position() is not None]
        if at_frontier and frontier_cells and len(visible_enemies) > 0:
            ref = min(visible_enemies, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            ex, ey = ref.get_position()
            ref_y = int(max(0, min(height - 1, ey)))
            target_pos = min(frontier_cells, key=lambda p: abs(p[1] - ref_y))
            best_dist = float('inf')
            best_actions = []
            for a in actions:
                if a == Directions.STOP:
                    continue
                succ = game_state.generate_successor(self.index, a)
                succ_pos = succ.get_agent_state(self.index).get_position()
                sx, sy = int(succ_pos[0]), int(succ_pos[1])
                if sx != frontier_x:
                    continue
                d = abs(sy - target_pos[1])
                if d < best_dist:
                    best_dist = d
                    best_actions = [a]
                elif d == best_dist:
                    best_actions.append(a)
            if best_actions:
                return random.choice(best_actions)
        if frontier_cells:
            target_pos = min(frontier_cells, key=lambda p: self.get_maze_distance(my_pos, p))
            best_dist = float('inf')
            best_actions = []
            for a in actions:
                if a == Directions.STOP:
                    continue
                succ = game_state.generate_successor(self.index, a)
                succ_pos = succ.get_agent_state(self.index).get_position()
                sx, sy = int(succ_pos[0]), int(succ_pos[1])
                d = self.get_maze_distance((sx, sy), target_pos)
                if d < best_dist:
                    best_dist = d
                    best_actions = [a]
                elif d == best_dist:
                    best_actions.append(a)
            if best_actions:
                return random.choice(best_actions)
        non_stop_actions = [a for a in actions if a != Directions.STOP]
        if non_stop_actions:
            return random.choice(non_stop_actions)
        return Directions.STOP

    def get_weights(self, game_state, action):
        score = self.get_score(game_state)
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        if self.red:
            invaders = [
                e for e in enemies
                if e.is_pacman and e.get_position() is not None and e.get_position()[0] <= mid_x - 1
            ]
        else:
            invaders = [
                e for e in enemies
                if e.is_pacman and e.get_position() is not None and e.get_position()[0] >= mid_x
            ]
        there_are_invaders = len(invaders) > 0
        if there_are_invaders:
            return {
                'num_invaders': -50000,
                'on_defense': 300,
                'invader_distance': -1000,
                'on_invader': 20000,
                'stop': -500,
                'reverse': -10,
                'distance_to_defended_food': -0.01,
                'distance_to_food': -0.05,
                'carrying': 0.5,
                'distance_to_frontier': -0.01
            }
        else:
            if score >= 2:
                return {
                    'num_invaders': -10000,
                    'on_defense': 200,
                    'invader_distance': -100,
                    'on_invader': 5000,
                    'stop': -100,
                    'reverse': -4,
                    'distance_to_defended_food': -2,
                    'distance_to_food': -3,
                    'carrying': 4,
                    'distance_to_frontier': -0.3
                }
            else:
                return {
                    'num_invaders': -10000,
                    'on_defense': 120,
                    'invader_distance': -100,
                    'on_invader': 5000,
                    'stop': -100,
                    'reverse': -2,
                    'distance_to_defended_food': -1,
                    'distance_to_food': -4,
                    'carrying': 5,
                    'distance_to_frontier': -0.2
                }
