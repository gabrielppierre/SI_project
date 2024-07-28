import numpy as np
import connection

# Parâmetros do Q-Learning
alpha = 0.1
gamma = 0.9
initial_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995
num_episodes = 1000  # Defina o número de episódios

# Inicialização da Q-Table
num_states = 96
num_actions = 3
Q = np.zeros((num_states, num_actions))

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

def get_initial_state(socket):
    # Envia uma ação inicial para obter o estado inicial
    initial_action = 'jump'
    state, _ = connection.get_state_reward(socket, initial_action)
    return state

def check_goal_state(state):
    # Atualize esses valores conforme você identificar as plataformas corretas
    platform_goals = ['10111', '01101']  # Plataformas 23 e 13
    direction_goal = '00'  # Norte

    # Extrai a plataforma e a direção do estado atual
    platform = state[:5]
    direction = state[5:]

    # Verifica se o estado atual é um dos estados objetivos
    return platform in platform_goals and direction == direction_goal

def save_q_table(Q):
    np.savetxt('q_table.txt', Q, fmt='%.6f')

def state_to_index(state):
    """
    Converte um estado binário para um índice inteiro.
    """
    return int(state, 2)

def main():
    socket = connection.connect(port=2037)
    if socket == 0:
        print("Falha na conexão. Encerrando o programa.")
        return

    epsilon = initial_epsilon
    for episode in range(num_episodes):
        state = get_initial_state(socket)
        done = False
        while not done:
            action = choose_action(state, epsilon)
            action_str = ['left', 'right', 'jump'][action]
            next_state, reward = connection.get_state_reward(socket, action_str)
            update_q_table(state, action, reward, next_state)
            state = next_state
            if check_goal_state(state):
                done = True
        epsilon = max(min_epsilon, epsilon * decay_rate)
    save_q_table(Q)

if __name__ == "__main__":
    main()
