import os
import numpy as np
import matplotlib.pyplot as plt
import connection

# Parâmetros do Q-Learning
alpha = 0.1
gamma = 0.9
initial_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995
num_episodes = 1000
max_steps_per_episode = 100
save_interval = 10
convergence_threshold = 1e-4

# Inicialização da Q-Table
num_states = 24 * 4  # 24 plataformas * 4 direções possíveis
num_actions = 3
Q = np.zeros((num_states, num_actions))

# Métricas para monitorar o treinamento
rewards_per_episode = []
epsilon_values = []
td_errors_per_episode = []
rewards_per_step = []  # Adicionando lista para recompensas por passo

def choose_action(state, epsilon):
    state_index = state_to_index(state)
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2])  # 0: left, 1: right, 2: jump
    else:
        return np.argmax(Q[state_index])

def update_q_table(state, action, reward, next_state):
    state_index = state_to_index(state)
    next_state_index = state_to_index(next_state)
    best_next_action = np.argmax(Q[next_state_index])
    td_target = reward + gamma * Q[next_state_index][best_next_action]
    td_error = td_target - Q[state_index][action]
    Q[state_index][action] += alpha * td_error
    return np.abs(td_error)

def get_initial_state(socket):
    initial_action = 'jump'
    state, _ = connection.get_state_reward(socket, initial_action)
    return state

def check_goal_state(state):
    platform_bits = state[2:7]  # Supondo que os bits 2 a 6 representam a plataforma
    platform_int = int(platform_bits, 2)  # Converte para inteiro
    platform_goals = [23, 13]  # Plataformas 23 e 13 em decimal
    
    print(f'Plataforma atual (binária): {platform_bits}')
    print(f'Plataforma atual (inteiro): {platform_int}')
    
    return platform_int in platform_goals

def save_q_table(Q, filename='q_table.txt'):
    np.savetxt(filename, Q, fmt='%.6f')

def state_to_index(state):
    platform_bits = state[2:7]  # Extrai os 5 bits da plataforma
    direction_bits = state[-2:]  # Extrai os 2 bits da direção
    return int(platform_bits + direction_bits, 2)  # Combina plataforma e direção

def plot_metrics(episodes, rewards, epsilons, td_errors, rewards_step, interval):
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.plot(episodes, rewards, label='Recompensa Acumulada')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa por Episódio')
    plt.legend()
    plt.savefig(f'graficos/recompensa_por_eposodio_{interval}.png')

    plt.subplot(1, 4, 2)
    plt.plot(episodes, epsilons, label='Epsilon')
    plt.xlabel('Episódios')
    plt.ylabel('Epsilon')
    plt.title('Taxa de Exploração')
    plt.legend()
    plt.savefig(f'graficos/taxa_de_exploracao_{interval}.png')

    plt.subplot(1, 4, 3)
    plt.plot(episodes, td_errors, label='Erro TD')
    plt.xlabel('Episódios')
    plt.ylabel('Erro TD')
    plt.title('Erro Total de TD por Episódio')
    plt.legend()
    plt.savefig(f'graficos/erro_td_por_eposodio_{interval}.png')

    plt.subplot(1, 4, 4)
    plt.plot(np.concatenate([np.full(max_steps_per_episode, episode) for episode in range(len(rewards_step))]),
             np.concatenate(rewards_step), label='Recompensas por Passo')
    plt.xlabel('Passos')
    plt.ylabel('Recompensa')
    plt.title('Recompensa por Passo')
    plt.legend()
    plt.savefig(f'graficos/recompensa_por_passo_{interval}.png')

    plt.tight_layout()
    plt.close()

def main():
    socket = connection.connect(port=2037)
    if socket == 0:
        print("Falha na conexão. Encerrando o programa.")
        return

    epsilon = initial_epsilon
    for episode in range(num_episodes):
        state = get_initial_state(socket)
        done = False
        total_td_error = 0
        total_reward = 0
        episode_rewards = []  # Armazenar recompensas por passo

        for step in range(max_steps_per_episode):
            if done:
                break
            action = choose_action(state, epsilon)
            action_str = ['left', 'right', 'jump'][action]
            next_state, reward = connection.get_state_reward(socket, action_str)
            td_error = update_q_table(state, action, reward, next_state)
            total_td_error += td_error
            total_reward += reward
            episode_rewards.append(reward)  # Adicionar recompensa do passo

            state = next_state
            if check_goal_state(state):
                done = True

        rewards_per_episode.append(total_reward)
        epsilon_values.append(epsilon)
        td_errors_per_episode.append(total_td_error)
        rewards_per_step.append(episode_rewards)  # Adicionar recompensas por passo ao final do episódio

        if (episode + 1) % save_interval == 0:
            plot_metrics(range(len(rewards_per_episode)), rewards_per_episode, epsilon_values, td_errors_per_episode, rewards_per_step, episode + 1)
        
        if total_td_error < convergence_threshold:
            print(f"Convergiu após {episode + 1} episódios.")
            break

        epsilon = max(min_epsilon, epsilon * decay_rate)

    save_q_table(Q)
    episodes = range(len(rewards_per_episode))
    plot_metrics(episodes, rewards_per_episode, epsilon_values, td_errors_per_episode, rewards_per_step, 'final')

if __name__ == "__main__":
    main()
