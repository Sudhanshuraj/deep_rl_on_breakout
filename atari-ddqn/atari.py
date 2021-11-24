import gym
import argparse
import numpy as np
import atari_py
import copy
import cv2
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from game_models.ge_game_model import GETrainer, GESolver
# from gym_wrappers import MainGymWrapper
import pdb
import os


FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
# INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)
INPUT_SHAPE= (37,165,1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

colors = [87, 103, 106, 126, 124, 110, 139]
ball_color = paddle_color = 87 # [200, 72, 72]
paddle_height = 4
brick_height = 6
brick_upper_limit = 57
brick_lower_limit = 94
paddle_upper_limit = 188
paddle_lower_limit = 193
ball_offset = 8
window_height = 210
window_width = 160
border_width = 8
ball_init_up = 90
ball_init_down = 130
ball_init_left = 60
ball_init_right = 90
ball_pos = []
ball_velocity = [0,0]
rel_ball_pos = [0,0]
prev_lives = 5
def extract_ball_pos(img, priv_ball_pos):
    new_ball_pos = []
    priv_ball_pos_up =  priv_ball_pos[0][0]
    priv_ball_pos_down =  priv_ball_pos[-1][0]
    priv_ball_pos_left =  priv_ball_pos[0][1]
    priv_ball_pos_right =  priv_ball_pos[-1][1]
    for x in range(priv_ball_pos_up-ball_offset, priv_ball_pos_down+ball_offset):
        for y in range(priv_ball_pos_left - ball_offset, priv_ball_pos_right + ball_offset):
            c = img[x][y]
            present_color = c # [c[0], c[1], c[2]]
            try:
                color_left = img[x][y-2] # [img[x][y-2][0], img[x][y-2][1], img[x][y-2][2]]
                color_right = img[x][y+2] # [img[x][y+2][0], img[x][y+2][1], img[x][y+2][2]]
                color_up = img[x-2][y] # [img[x-2][y][0], img[x-2][y][1], img[x-2][y][2]]
                color_down = img[x+2][y] # [img[x+2][y][0], img[x+2][y][1], img[x+2][y][2]]
            except Exception as e:
                pass
            if color_left not in colors and color_right not in colors and present_color == ball_color:
                new_ball_pos.append([x,y])
    return new_ball_pos

def extract_paddle_pos(img):
    paddle_pos_raw = img[paddle_upper_limit:paddle_lower_limit]
    paddle_pos = []
    for x in range(paddle_upper_limit, paddle_lower_limit):
        for y in range(border_width, paddle_pos_raw.shape[1]-border_width):
            c = img[x][y]
            present_color = c # [c[0], c[1], c[2]]
            if present_color == paddle_color:
                try:
                    color_left = img[x][y-2] # [img[x][y-2][0], img[x][y-2][1], img[x][y-2][2]]
                    color_right = img[x][y+2] # [img[x][y+2][0], img[x][y+2][1], img[x][y+2][2]]
                    # color_up = img[x-2][y] # [img[x-2][y][0], img[x-2][y][1], img[x-2][y][2]]
                    # color_down = img[x+2][y] # [img[x+2][y][0], img[x+2][y][1], img[x+2][y][2]]
                except Exception as e:
                    pass
                if color_left not in colors and color_right not in colors:
                    pass
                else:
                    paddle_pos.append([x,y])
    return paddle_pos

def extract_init_ball_pos(img):
    ball_pos = []
    for x in range(ball_init_up, ball_init_down):
        for y in range(ball_init_left, ball_init_right):
            c = img[x][y]
            present_color = c # [c[0], c[1], c[2]]
            if present_color in colors:
                try:
                    color_left = img[x][y-2] # [img[x][y-2][0], img[x][y-2][1], img[x][y-2][2]]
                    color_right = img[x][y+2] # [img[x][y+2][0], img[x][y+2][1], img[x][y+2][2]]
                    # color_up = img[x-2][y] # [img[x-2][y][0], img[x-2][y][1], img[x-2][y][2]]
                    # color_down = img[x+2][y] # [img[x+2][y][0], img[x+2][y][1], img[x+2][y][2]]
                except Exception as e:
                    pass
                if color_left not in colors and color_right not in colors and present_color == ball_color:
                    ball_pos.append([x,y])

    return np.asarray(ball_pos)

def feature_vector(gray):   
    # env.render()
    global ball_pos,ball_velocity,rel_ball_pos,priv_features
    bricks_pos = gray[brick_upper_limit:brick_lower_limit]
    paddle_pos = extract_paddle_pos(gray)
    avg_paddle_pos = np.average(paddle_pos, axis=0)
    final_feature_vector = ((np.asarray(bricks_pos)).flatten() != 0)*1
    if len(ball_pos) == 0:
        present_ball_pos = extract_init_ball_pos(gray)
        if len(present_ball_pos) > 0:
            ball_pos = present_ball_pos
            rel_ball_pos = (np.average(ball_pos, axis=0) - avg_paddle_pos)
            ball_velocity = [0,0]
    else:
        new_ball_pos = extract_ball_pos(gray, ball_pos)
        if len(new_ball_pos) > 0: 
            ball_velocity = [new_ball_pos[0][0] - ball_pos[0][0], new_ball_pos[0][1] - ball_pos[0][1]]
        else:
            ball_velocity = [0,0]
        ball_pos = copy.deepcopy(new_ball_pos)
        rel_ball_pos = (np.average(ball_pos, axis=0) - avg_paddle_pos)

    final_feature_vector = list(final_feature_vector)
    final_feature_vector.append(avg_paddle_pos[1])
    if rel_ball_pos[0]>=0  or rel_ball_pos[0] < 0:
        final_feature_vector.append(rel_ball_pos[0]+210)
        final_feature_vector.append(rel_ball_pos[1]+210)
    else :
        final_feature_vector.append(210)
        final_feature_vector.append(210)
    final_feature_vector.append(ball_velocity[0]+10)
    final_feature_vector.append(ball_velocity[1]+10)
    # pdb.set_trace()
    final_feature_vector_2D = np.zeros((37,165))
    final_feature_vector_2D[:,:-5]=np.array(final_feature_vector[:-5]).reshape(37,160)
    final_feature_vector_2D[:,-1:]=np.zeros((37,1))+final_feature_vector[-1]
    final_feature_vector_2D[:,-2:-1]=np.zeros((37,1))+final_feature_vector[-2]
    final_feature_vector_2D[:,-3:-2]=np.zeros((37,1))+final_feature_vector[-3]
    final_feature_vector_2D[:,-4:-3]=np.zeros((37,1))+final_feature_vector[-4]
    final_feature_vector_2D[:,-5:-4]=np.zeros((37,1))+final_feature_vector[-5]
    return final_feature_vector_2D.astype(np.float64).reshape(37,165,1)

x_dist = []
# def get_extra_reward(info):
#     global prev_lives,x_dist
#     reward = 0
#     if prev_lives > info:
#         if len(x_dist) > 0 :
#             reward -= abs(x_dist[-1]-210)
#             if len(x_dist)>1:
#                 if (abs(x_dist[-2]-120) - abs(x_dist[-1]-120)) > 0:
#                     reward +=10
#                 elif (abs(x_dist[-2]-120) - abs(x_dist[-1]-120)) < 0:
#                     reward -=10 
#         prev_lives-=1
#         reward-=80
#         x_dist = []
#     else:
#         reward+=1

#     if prev_lives == 0:
#         prev_lives=5
#         reward-=100
    
#     return reward
 
def get_extra_reward(info):
    global prev_lives, x_dist
    reward = 0
    if prev_lives > info:
        if len(x_dist) > 0 :
            reward -= abs(x_dist[-1])/210
            if len(x_dist)>1:
                if (abs(x_dist[-2]) - abs(x_dist[-1])) > 0:
                    reward -= 0.1
                elif (abs(x_dist[-2]) - abs(x_dist[-1])) < 0:
                    reward -= 0.2 
        prev_lives-=1
        reward-=2
        x_dist = []
    else:
        reward+= 0.01

    if prev_lives == 0:
        prev_lives=5
        reward-=5
    
    return reward

start_from_0 = False
init_steps = 300000
pre_trained_steps = 450000
import time
class Atari:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_run_limit, clip = self._args()
        env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
        env = gym.make(env_name)
        # env = MainGymWrapper.wrap(env1)
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env, render, total_step_limit, total_run_limit, clip)

    def _main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        if isinstance(game_model, GETrainer):
            game_model.genetic_evolution(env)

        global x_dist

        run = 0
        total_step = max(0,init_steps)
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            rgb = env.reset()
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            current_state = feature_vector(gray)
            # pdb.set_trace()
            step = 0
            score = 0
            mini = 999999
            maxi = -999999
            prev_lives = 5
            x_dist=[]
            while True:
                if total_step >= total_step_limit:
                    print("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                # print("action",action)
                rgb, reward, terminal, info = env.step(action)
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                next_state = feature_vector(gray)
                # if clip:
                    # np.sign(reward)
                score += reward

                if not current_state[0,-3,0]==210 :
                    x_dist.append(current_state[0,-3,0])

                reward+=get_extra_reward(info['ale.lives'])
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state
                # pdb.set_trace()

                game_model.step_update(total_step)  
                if terminal:
                    game_model.save_run(score, step, run)
                    break
            # print("                                                    run,total_step,step",run,total_step,step)


    def _args(self):
        parser = argparse.ArgumentParser()
        available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
        parser.add_argument("-g", "--game", help="Choose from available games: " + str(available_games) + ". Default is 'breakout'.", default="Breakout")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test, ge_train, ge_test. Default is 'ddqn_training'.", default="ddqn_training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'", default=True, type=bool)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        print("Selected game: " + str(game_name))
        print("Selected mode: " + str(game_mode))
        print("Should render: " + str(render))
        print("Should clip: " + str(clip))
        print("Total step limit: " + str(total_step_limit))
        print("Total run limit: " + str(total_run_limit))
        return game_name, game_mode, render, total_step_limit, total_run_limit, clip

    def _game_model(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_training":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space,start_from_0,steps=init_steps)
        elif game_mode == "ddqn_testing":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ge_training":
            return GETrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ge_testing":
            return GESolver(game_name, INPUT_SHAPE, action_space)
        else:
            print("Unrecognized mode. Use --help")
            exit(1)


if __name__ == "__main__":
    Atari()
