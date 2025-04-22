import gymnasium as gym
import numpy as np
import random
from envs.swimmer import Swimmer
from envs.half_cheetah import HalfCheetah
from envs.lq import LQ
from envs.minigolf import MiniGolf
import json
import io
from tqdm import tqdm
import datetime
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--trial",
    help="Number of trial.",
    type=int,
    default=1
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="swimmer",
    choices=["swimmer", "half_cheetah", "lq", "minigolf"]
)
parser.add_argument(
    "--horizon",
    help="The horizon amount.",
    type=int,
    default=200
)
parser.add_argument(
    "--gamma",
    help="The discount factor.",
    type=float,
    default=0.995
)
parser.add_argument(
    "--clip",
    help="Clip the action.",
    default=False,
    action='store_true'
)
parser.add_argument(
    "--dir",
    help="Directory to save the results.",
    type=str,
    default=""
)

# === Nodo ===
class Node:
    def __init__(self, id, node_type):
        self.id = id
        self.type = node_type  # 'input', 'hidden', 'output'
        self.value = 0.0

# === Connessione ===
class Connection:
    def __init__(self, in_node, out_node, weight, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled

# === Rete Evolvibile ===
class EvolvableNetwork:
    def __init__(self, input_size, output_size):
        self.nodes = []
        self.connections = []
        self.input_size = input_size
        self.output_size = output_size
        self.node_id_counter = 0

        # Crea nodi input
        for _ in range(input_size):
            self.nodes.append(Node(self._next_id(), 'input'))

        # Crea nodi output
        for _ in range(output_size):
            self.nodes.append(Node(self._next_id(), 'output'))

        # Connessioni random iniziali input → output
        input_nodes = self.get_nodes_by_type('input')
        output_nodes = self.get_nodes_by_type('output')
        for inp in input_nodes:
            for out in output_nodes:
                # weight = np.random.randn()
                weight = 0
                self.connections.append(Connection(inp.id, out.id, weight))

    def _next_id(self):
        self.node_id_counter += 1
        return self.node_id_counter

    def get_nodes_by_type(self, node_type):
        return [n for n in self.nodes if n.type == node_type]

    def forward(self, inputs):
        node_outputs = {n.id: 0.0 for n in self.nodes}

        # Set input values
        input_nodes = self.get_nodes_by_type('input')
        for i, n in enumerate(input_nodes):
            node_outputs[n.id] = inputs[i]

        # Propagazione manuale per piccoli grafi
        for conn in self.connections:
            if conn.enabled:
                node_outputs[conn.out_node] += node_outputs[conn.in_node] * conn.weight

        # Attivazione output con tanh
        outputs = []
        for n in self.get_nodes_by_type('output'):
            outputs.append(np.tanh(node_outputs[n.id]))
        return np.array(outputs)

    def mutate_weights(self, rate=0.8, strength=0.5):
        for conn in self.connections:
            if random.random() < rate:
                conn.weight += np.random.randn() * strength

    def mutate_add_connection(self):
        possible_in = self.nodes
        possible_out = [n for n in self.nodes if n.type != 'input']
        in_node = random.choice(possible_in)
        out_node = random.choice(possible_out)
        if in_node.id == out_node.id:
            return  # no self-loop
        # Evita duplicati
        for conn in self.connections:
            if conn.in_node == in_node.id and conn.out_node == out_node.id:
                return
        self.connections.append(Connection(in_node.id, out_node.id, np.random.randn()))

    def mutate_add_node(self):
        conn = random.choice([c for c in self.connections if c.enabled])
        conn.enabled = False
        new_node = Node(self._next_id(), 'hidden')
        self.nodes.append(new_node)

        # Due nuove connessioni
        self.connections.append(Connection(conn.in_node, new_node.id, 1.0))
        self.connections.append(Connection(new_node.id, conn.out_node, conn.weight))

    def clone(self):
        new_net = EvolvableNetwork(self.input_size, self.output_size)
        new_net.nodes = [Node(n.id, n.type) for n in self.nodes]
        new_net.connections = [Connection(c.in_node, c.out_node, c.weight, c.enabled) for c in self.connections]
        new_net.node_id_counter = self.node_id_counter
        return new_net

    def print_summary(self):
        total_nodes = len(self.nodes)
        input_nodes = len(self.get_nodes_by_type('input'))
        hidden_nodes = len(self.get_nodes_by_type('hidden'))
        output_nodes = len(self.get_nodes_by_type('output'))
        total_connections = len([c for c in self.connections if c.enabled])
        
        total_params = (input_nodes * hidden_nodes) + (hidden_nodes * output_nodes) 

        print("=== Network Summary ===")
        print(f"Total Nodes       : {total_nodes}")
        print(f"  - Input Nodes   : {input_nodes}")
        print(f"  - Hidden Nodes  : {hidden_nodes}")
        print(f"  - Output Nodes  : {output_nodes}")
        print(f"Active Connections: {total_connections}")
        print(f"Total Parameters  : {total_params}")

        return total_params


# === Fitness ===
def evaluate(env, network, episodes=1):
    tot_reward = 0
    perf = np.zeros(episodes, dtype=np.float64)
    for e in range(episodes):
        if(args.env != "lq" and args.env != "minigolf"):
            obs, _ = env.reset()
        else:
            obs = env.reset()
        for h in range(env.horizon):
            action = network.forward(obs)
            obs, reward, _, _ = env.step(action)
            tot_reward += (env.gamma ** h) * reward
        perf[e] = tot_reward
        tot_reward = 0
    if(args.env != "lq" and args.env != "minigolf"):
        env.close()
    return np.mean(perf)  # media dei reward per episodio

# === Evoluzione ===
def run_evolution(env, dir_name, population_size=5, generations=5):
    print(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    input_size = env.state_dim
    output_size = env.action_dim
    if(args.env != "lq" and args.env != "minigolf"):
        env.close()
    performance = np.zeros(generations, dtype=np.float64)

    population = [EvolvableNetwork(input_size, output_size) for _ in range(population_size)]

    for gen in tqdm(range(generations)):
        fitness = [evaluate(env, net, episodes=200) for net in population]
        best_idx = np.argmax(fitness)
        # print(f"Gen {gen:03d} | Best fitness: {fitness[best_idx]:.2f}")

        # Selezione: top 10%
        sorted_indices = np.argsort(fitness)[::-1]
        survivors = [population[i] for i in sorted_indices[:max(1, population_size // 10)]]

        # Evaluate the best
        performance[gen] = evaluate(env, survivors[0], episodes=200)

        # Ripopolamento: mutazioni dei sopravvissuti
        new_population = []
        while len(new_population) < population_size:
            parent = random.choice(survivors)
            child = parent.clone()
            if random.random() < 0.8:
                child.mutate_weights()
            if random.random() < 0.3:
                child.mutate_add_connection()
            if random.random() < 0.2:
                child.mutate_add_node()
            new_population.append(child)

        population = new_population
        
        # Salva i risultati
        if gen % 10 == 0:
            save_results(dir_name, performance)

    # Visualizzazione
    best_net = population[best_idx]
    # input("Premi invio per visualizzare la migliore rete...")
    # visualize(best_net, env)
    save_results(dir_name, performance)
    return best_net

def visualize(network, env):
    if(args.env != "lq" and args.env != "minigolf"):
        obs, _ = env.reset()
    else:
        obs = env.reset()
    total_reward = 0
    for h in range(env.horizon):
        action = network.forward(obs)
        obs, reward, _, _ = env.step(action)
        total_reward += (env.gamma ** h) * reward
    if(args.env != "lq" and args.env != "minigolf"):
        env.close()
    print(f"Reward ottenuto: {total_reward:.2f}")

def save_results(dir_name, performance):
    results = {"performance": np.array(performance, dtype=float).tolist()}
    name = dir_name + "/results.json"
    with io.open(name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
        f.close()

# === Avvio ===
if __name__ == "__main__":
    args = parser.parse_args()
    dir_name = args.dir
    params = np.zeros(args.trial, dtype=np.float64)
    
    dir_name += f"{args.env}/"
    dir_name += datetime.datetime.now().strftime("%m_%d-%H_%M") + "/"
    dir_name += f"H_{args.horizon}_g_{str(args.gamma).replace('.', '')}"
    
    for i in range(args.trial):
        if args.env == "swimmer":
            env_class = Swimmer
            env = Swimmer(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        elif args.env == "half_cheetah":
            env_class = HalfCheetah
            env = HalfCheetah(horizon=args.horizon, gamma=args.gamma, clip=bool(args.clip))
        elif args.env == "lq":
            env_class = LQ
            env = LQ(horizon=args.horizon, gamma=args.gamma)
        elif args.env == "minigolf":
            env_class = MiniGolf
            env = MiniGolf(horizon=args.horizon, gamma=args.gamma)


        save_dir = dir_name + f"/trial_{i}"

        best = run_evolution(env, save_dir, population_size=5, generations=100)
        params[i] = best.print_summary()

    name = dir_name + "/params.json"
    par = { "params": np.array(params, dtype=float).tolist()}
    with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(par, ensure_ascii=False, indent=4))
            f.close()


