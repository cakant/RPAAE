
from collections import namedtuple
from itertools import count
import math
import random

from wrappers import *
from memory import ReplayMemory
from RPAAEmodels import *

import torch
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def optimize_model(batch):
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8).bool()
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def Eoptimize_model(batch):
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: 0 * torch.tensor([r], device='cuda'), batch.reward)))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8).bool()
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    state_action_values = Epolicy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = Etarget_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    eoptimizer.zero_grad()
    loss.backward()
    for param in Epolicy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    eoptimizer.step()


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def train(env, n_episodes, RPAAE):
    if RPAAE:
        f = open('attack.txt', 'a')
    else:
        f = open('no-attack.txt', 'a')
    attack = 0
    step = 0
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            step += 1
            action = select_action(state)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = get_state(obs)
                Qsn = policy_net(next_state.to('cuda')).data.cpu().numpy().ravel()
            else:
                next_state = None
                Qsn = [0]
            Qs = policy_net(state.to('cuda')).data.cpu().numpy().ravel()
            if RPAAE:
                Es = Epolicy_net(state.to('cuda')).data.cpu().numpy().ravel()
            if RPAAE and Es[np.argmax(Qs)] != min(Es):
                dt = min(((max(Qs) - (1 - (1e-4)) * Qs[np.argmin(Es)]) / (1e-4) - 0.99 * max(Qsn)) * (1 + 0.8),
                         abs(max(Qs)) / 2)
                if action[0][0] == np.argmin(Es):
                    attack += 1
                    reward += dt
                else:
                    attack += 1
                    reward -= dt
                reward = np.float(reward)
            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state
            if steps_done > INITIAL_MEMORY:
                if len(memory) > BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    optimize_model(batch)
                    if episode < n_episodes * 0.03 and RPAAE:
                        Eoptimize_model(batch)
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    if episode < n_episodes * 0.03 and RPAAE:
                        Etarget_net.load_state_dict(Epolicy_net.state_dict())
            if done:
                break
        f.write(str(total_reward) + '\n')
        if RPAAE:
            print('Episode: {}\t Attack-Frequency: {}\t Total reward: {}'.format(episode,attack / step, total_reward,))
        else:
            print('Episode: {} \t Total reward: {} '.format(episode, total_reward,))
    env.close()
    f.close()
    return


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RPAAE = True
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    if RPAAE:
        Epolicy_net = EDQN(n_actions=4).to(device)
        for p in Epolicy_net.parameters():
            p.data.fill_(0)
        Etarget_net = EDQN(n_actions=4).to(device)
        for p in Etarget_net.parameters():
            p.data.fill_(0)
        Etarget_net.load_state_dict(Epolicy_net.state_dict())
        eoptimizer = optim.Adam(Epolicy_net.parameters(), lr=lr)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    memory = ReplayMemory(MEMORY_SIZE)
    train(env, 2500, RPAAE)

