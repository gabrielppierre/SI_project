import connection as cn
import numpy as np
import random

learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1
num_episodes = 1000

q_table = np.zeros((96, 3))

def state_to_index(state):
    platform = int(state[:5], 2)
    direction = int(state[5:], 2)
    return platform * 4 + direction 
def choose_action(state_index):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 2)
    else:
        return np.argmax(q_table[state_index])

def main():
    port = 2037

    s = cn.connect(port)

    if s == 0:
        print("falha ao estabelecer a conexao")
        return
    print("conexao estabelecida com sucesso")

    for episode in range(num_episodes):

        state, _ = cn.get_state_reward(s, "jump")
        state_index = state_to_index(state)
        done = False
        steps = 0

        while not done:
            action = choose_action(state_index)
            action_str = ["left", "right", "jump"][action]

            new_state, reward = cn.get_state_reward(s, action_str)
            new_state_index = state_to_index(new_state)

            best_next_action = np.argmax(q_table[new_state_index])
            td_target = reward + discount_factor * q_table[new_state_index, best_next_action]
            td_error = td_target - q_table[state_index, action]
            q_table[state_index, action] += learning_rate * td_error

            state_index = new_state_index

            steps += 1

            if new_state == "terminal" or reward == -14:  # -14 é a recompensa de perder
                done = True

        print(f"episodio {episode + 1} concluido em {steps} passos")

    np.savetxt("q_table.txt", q_table, fmt='%f')

if __name__ == "__main__":
    main()
