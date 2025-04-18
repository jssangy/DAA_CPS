import yaml
import wandb
import numpy as np
from tqdm import tqdm

from Environment import ENV
from Agent import DQN_Agent

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(cfg):
    # wandb 설정
    wandb.init(project='DAA_CPS', name='DQN_training')
    wandb.config.update(cfg['params'])
    
    env_test = ENV()

    # 파라미터 설정
    state_size = env_test.get_state().size
    hidden_size = cfg['params']['hidden_size']
    action_size = len(cfg['params']['action'])
    learning_rate = cfg['params']['learning_rate']
    gamma = cfg['params']['gamma']
    memory_size = cfg['params']['memory_size']
    episodes = cfg['params']['episode']
    batch_size = cfg['params']['batch_size']
    target_update_frequency = cfg['params']['target_update_frequency']
    timesteps  = cfg['params']['timestep']
    reward = cfg['params']['reward']
    
    # 에이전트 초기화
    agent = DQN_Agent(state_size, hidden_size, action_size, learning_rate, gamma, memory_size)    
    
    # 훈련 모니터링을 위한 변수들
    best_reward = float('-inf')
    best_episode = 0
    
    print("Train Begin...")
    
    for episode in tqdm(range(episodes), desc="Episodes"):
        # 매 에피소드마다 환경 초기화
        env = ENV()
        state = np.array(env.get_state().flatten())
        
        total_reward = 0
        episode_losses = []
        
        for timestep in tqdm(range(timesteps), desc=f'Episode {episode}', leave=False):
            # State & Action
            actions = {}
            for num in env.controller.agv_nums:
                if timestep == 0 or not any(events[num]):
                    actions[num] = 0
                else:
                    actions[num] = agent.act(state, num)
            
            # Reward & Next State
            next_state, reward, events = env.step(actions, reward)
            
            next_state = np.array(next_state.flatten())

            total_reward += reward

            actions_list = list(actions.values())
            agent.remember(state, actions_list, reward, next_state)
            agent.replay(batch_size)

            loss = agent.get_loss()
            episode_losses.append(loss)

            state = next_state

        avg_loss_episode = np.mean(episode_losses)
        # WandB 에피소드 단위로 기록
        wandb.log({
            'episode_avg_loss': avg_loss_episode,
            'episode_total_reward': total_reward
        })
        
        # 타겟 네트워크 업데이트
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # 최적 모델 저장
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            agent.save(f"{cfg['paths']['model']}best.pth")
        
        # 주기적인 모델 저장
        if episode % 100 == 0:
            agent.save(f"{cfg['paths']['model']}{episode}.pth")
        
        # 훈련 진행 상황 출력
        print(f"\nReward: {total_reward}")
        print(f"Avg Loss: {avg_loss_episode}\n")
    
    print("Train End!")
    print(f"Best Reward: {best_reward:.2f} (Episode {best_episode})")
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    config = load_config('config.yaml')
    train(config)