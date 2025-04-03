import yaml
import wandb
import numpy as np
import torch
from collections import deque

from Environment import ENV
from Agent import Agent

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(cfg):
    # wandb 설정
    wandb.config.update(cfg['params'])
    
    env = ENV()
    # 환경 설정
    state_size = env.get_state().size
    hidden_size = cfg['params']['hidden_size']
    action_size = len(cfg['params']['action'])
    learning_rate = cfg['params']['learning_rate']
    gamma = cfg['params']['gamma']
    memory_size = cfg['params']['memory_size']
    
    # 에이전트 초기화
    agent = Agent(state_size, hidden_size, action_size, learning_rate, gamma, memory_size)
    
    # 훈련 파라미터
    episodes = cfg['params']['episode']
    batch_size = cfg['params']['batch_size']
    target_update_frequency = cfg['params']['target_update_frequency']
    
    # 훈련 모니터링을 위한 변수들
    best_reward = int('-inf')
    best_episode = 0
    reward_history = deque(maxlen=100)  # 최근 100 에피소드의 보상 기록
    patience = 100  # 성능 개선이 없을 때 기다리는 에피소드 수
    no_improvement_count = 0
    
    print("훈련 시작...")
    
    for episode in range(episodes):
        # 매 에피소드마다 새로운 환경 생성
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # 행동 선택 및 실행
            action_idx = agent.act(state)
            action_name = cfg['params']['action'][action_idx]
            
            # 환경 스텝 진행
            next_state, reward, done = env.Run()
            next_state = np.reshape(next_state, [1, state_size])
            
            # 경험 저장 및 학습
            agent.remember(state, action_idx, reward, next_state, done)
            agent.replay(batch_size)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # wandb에 q_value와 loss만 기록
            stats = agent.get_statistics()
            wandb.log({
                'loss': stats['current_loss'],
                'q_value': stats['current_q_value'],
                'episode': episode
            }, step=episode)
            
            if steps % 1000 == 0:  # 메모리 관리
                torch.cuda.empty_cache()
        
        # 타겟 네트워크 업데이트
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # 보상 기록 및 모니터링
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history)
        
        # 최적 모델 저장
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            agent.save(f"{cfg['paths']['model']}best.pth")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 주기적인 모델 저장
        if episode % 100 == 0:
            agent.save(f"{cfg['paths']['model']}{episode}.pth")
        
        # 훈련 진행 상황 출력
        print(f"에피소드: {episode}/{episodes}")
        print(f"보상: {total_reward:.2f}")
        print(f"평균 보상 (최근 100): {avg_reward:.2f}")
        print(f"최고 보상: {best_reward:.2f} (에피소드 {best_episode})")
        print(f"단계 수: {steps}")
        print("-" * 50)
        
        # 조기 종료 조건
        if no_improvement_count >= patience:
            print(f"성능 개선이 {patience} 에피소드 동안 없어 훈련을 종료합니다.")
            break
    
    print("훈련 완료!")
    print(f"최고 보상: {best_reward:.2f} (에피소드 {best_episode})")
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    wandb.init(project='DAA_CPS', name='DQN_training')
    config = load_config('config.yaml')
    train(config)