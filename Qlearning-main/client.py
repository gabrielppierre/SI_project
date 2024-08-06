import connection as cn
import numpy as np
import os
import logging

# Configuracao do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QLearningAgent:
    def __init__(self, alpha=0.7, gamma=0.95, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, actions=None, q_table_filename='resultado.txt', server_port=2037, save_interval=10):
        """
        Inicializa o agente Q-Learning com os parâmetros fornecidos.
        
        :param alpha: Taxa de aprendizado.
        :param gamma: Fator de desconto.
        :param epsilon: Taxa de exploracao inicial.
        :param epsilon_decay: Taxa de decaimento do epsilon.
        :param min_epsilon: Valor minimo para epsilon.
        :param actions: Lista de acoes possiveis.
        :param q_table_filename: Nome do arquivo da Q-table.
        :param server_port: Porta do servidor do jogo.
        :param save_interval: Intervalo de episódios para salvar a Q-table.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = actions if actions is not None else ['left', 'right', 'jump']
        self.q_table_path = os.path.join(os.path.dirname(__file__), q_table_filename)
        self.server_port = server_port
        self.save_interval = save_interval
        self.q_table = self.load_q_table()
        self.connection = self.connect_to_server()

    def connect_to_server(self):
        """Estabelece conexao com o servidor do jogo."""
        try:
            connection = cn.connect(self.server_port)
            return connection
        except Exception as e:
            raise

    def load_q_table(self):
        """Carrega a Q-table de um arquivo."""
        try:
            q_table = np.loadtxt(self.q_table_path)
            logging.info("Q-table carregada com sucesso.")
            return q_table
        except Exception as e:
            logging.error(f"Erro ao carregar a Q-table: {e}")
            raise

    def save_q_table(self):
        """Salva a Q-table em um arquivo."""
        try:
            np.savetxt(self.q_table_path, self.q_table)
            logging.info("Q-table salva com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao salvar a Q-table: {e}")
            raise

    def choose_action(self, state):
        """Escolhe uma acao com base na politica epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
            logging.debug(f"Acao escolhida aleatoriamente: {action}")
        else:
            action = self.actions[np.argmax(self.q_table[state, :])]
            logging.debug(f"Acao escolhida pela politica: {action}")
        return action

    def update_q_table(self, state, action, reward, next_state):
        """Atualiza a Q-table usando a formula do Q-learning."""
        action_index = self.actions.index(action)
        best_next_action = np.max(self.q_table[next_state, :])
        self.q_table[state, action_index] += self.alpha * (
            reward + self.gamma * best_next_action - self.q_table[state, action_index]
        )
        logging.debug(f"Q-table atualizada para o estado {state} e acao {action}")

    def get_initial_state(self):
        """Obtem o estado inicial do jogo."""
        state, reward = cn.get_state_reward(self.connection, "jump")
        platform = int(state[2:7], 2)
        direction = int(state[-2:], 2)
        logging.info(f'Plataforma: {platform}, Sentido: {direction}, Recompensa: {reward}')
        return int(state, 2), reward

    def train(self):
        """Loop principal de treinamento do agente."""
        state, _ = self.get_initial_state()
        episode_count = 0
        while True:
            episode_count += 1
            
            # Escolhe a acao baseada na politica epsilon-greedy
            action = self.choose_action(state)
            
            # Executa a acao e obtem o novo estado e recompensa
            next_state_str, reward = cn.get_state_reward(self.connection, action)
            next_state = int(next_state_str, 2)
            platform = int(next_state_str[2:7], 2)
            direction = int(next_state_str[-2:], 2)
            logging.info(f'Episodio: {episode_count}, Estado: {next_state}, Recompensa: {reward}, Plataforma: {platform}, Sentido: {direction}')
            
            # Atualiza a Q-table com base na acao tomada e a recompensa recebida
            self.update_q_table(state, action, reward, next_state)
            
            # Salva a Q-table a cada X episodios
            if episode_count % self.save_interval == 0:
                self.save_q_table()

            # Atualiza o estado atual
            state = next_state

            # Decai o epsilon para reduzir gradualmente a exploracao
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                logging.debug(f"Epsilon atualizado: {self.epsilon}")

if __name__ == "__main__":
    agent = QLearningAgent(save_interval=10)
    agent.train()
