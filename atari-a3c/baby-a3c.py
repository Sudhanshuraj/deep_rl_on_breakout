# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License
from __future__ import print_function
import cv2
import torch, os, gym, time, glob, argparse, sys
import numpy as np
import copy
from scipy.signal import lfilter
# from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'

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
x_dist = []

def get_extra_reward(info):
    global prev_lives, x_dist
    reward = 0

    if prev_lives > info:
        if len(x_dist) > 0 :
            # print("reward ",x_dist[-1])
            reward -= abs(x_dist[-1])/160
            if len(x_dist)>1:
                if (abs(x_dist[-2]) - abs(x_dist[-1])) > 0:
                    reward +=10/20
                elif (abs(x_dist[-2]) - abs(x_dist[-1])) < 0:
                    reward -=10/20 
        prev_lives-=1
        reward-=2/10
        x_dist = []

    if prev_lives == 0:
        prev_lives=5
        reward-=5/10
    
    return reward

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
        final_feature_vector.append(rel_ball_pos[0])
        final_feature_vector.append(rel_ball_pos[1])
    else :
        final_feature_vector.append(0)
        final_feature_vector.append(0)
    final_feature_vector.append(ball_velocity[0])
    final_feature_vector.append(ball_velocity[1])
    return final_feature_vector

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
# prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        input_shape = 5925
        output_shape1 = 1024
        output_shape2 = 256
        
        self.linear1 = nn.Linear(input_shape, output_shape1)
        self.linear2 = nn.Linear(output_shape1, output_shape2)

        # self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(output_shape2, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        output_shape2 = 256
        inputs, hx = inputs
        x = F.elu(self.linear1(inputs))
        x = F.elu(self.linear2(x))
        # x = F.elu(self.conv1(inputs))
        # x = F.elu(self.conv2(x))
        # x = F.elu(self.conv3(x))
        # x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, output_shape2), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args.env) # make a local (unshared) environment
    env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions) # a local/unshared model
    rgb = env.reset()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    current_state = feature_vector(gray)
    state = torch.tensor(current_state) # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= 8e7 or args.test: # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1,1,5925), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            rgb, reward, done, lives_info = env.step(action.numpy()[0])
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            current_state = feature_vector(gray)
          
            if args.render: env.render()

            state = torch.tensor(current_state) ; epr += reward

            if not current_state[-3]==210 :
                x_dist.append(current_state[-3])

            reward += get_extra_reward(lives_info['ale.lives'])
            done = done or episode_length >= 1e4 # don't playing one ep for too long
            
            info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0: # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                rgb = env.reset()
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                current_state = feature_vector(gray)
                state = torch.tensor(current_state)
               
            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        # mp.set_start_method('forkserver', force=True) # this must not be in global scope
        mp.set_start_method('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    
    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n # get the action space of this game
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    # info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()
