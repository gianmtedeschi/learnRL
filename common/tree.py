import numpy as np
import graphviz

class Node:
    def __init__(self, node_id, key=None, father=None):
        self.id_left = None
        self.id_right = None
        self.val = key # [value, split]
        self.node_id = node_id
        self.id_father = father


class BinaryTree:
    def __init__(self, nodes_num=1000):
        # array of all nodes
        self.nodes = np.array([Node(i) for i in range(nodes_num)])
        # array of boolean values, )True if the node is a leaf
        self.is_leaf = np.zeros(nodes_num, dtype=bool)
        # counter of used nodes in the nodes structure
        self.used = 0


    def insert_root(self, val):
        self.nodes[0].val = [val, None]
        self.is_leaf[self.used] = True
        self.used += 1

    def insert(self, param, id_father, split):
        # adjust split point in case of scalar problems
        if np.isscalar(split):
            split = [0, split]

        # update father node
        self.nodes[id_father].val[1] = split
        self.nodes[id_father].id_left = self.used
        self.nodes[id_father].id_right = self.used + 1
        self.is_leaf[id_father] = False

        # add son nodes
        # left
        self.nodes[self.used].id_father = id_father
        self.nodes[self.used].val = [param[0], None]
        self.is_leaf[self.used] = True

        # right
        self.nodes[self.used + 1].id_father = id_father
        self.nodes[self.used + 1].val = [param[1], None]
        self.is_leaf[self.used + 1] = True


        # increment used nodes by 2
        self.used += 2

    def update_all_leaves(self, param):
        i = 0

        if len(self.get_all_leaves()) != len(param):
            print("[TREE POLICY] Number of parameters is different from number of leaves", len(self.get_all_leaves()), len(param))
            return
        
        for Node in self.nodes:
            if self.is_leaf[Node.node_id]:
                Node.val[0] = param[i]
                i += 1
        
    def find_region_leaf(self, state, policy=False):
        # begin from the root
        current_node = 0

        while(self.is_leaf[current_node] == False):
            if self.find_direction(state, current_node, policy) == 0: 
                current_node = self.nodes[current_node].id_left
                # self.to_list(self.nodes[current_node])
            else:
                current_node = self.nodes[current_node].id_right
                # self.to_list(self.nodes[current_node])

        return self.nodes[current_node]

    def find_direction(self, state, current_node, policy):
        """
        Return the direction of the split based on the axis of the split point

        state = [axis, point]
        current_node = current node id

        return 0 if the state is on the left of the split point
        """
        split = self.nodes[current_node].val[1]
        if policy: 
            # Check if state is a scalar
            if len(state) == 1:
                state_value = state
            else:
                state_value = state[split[0]]

            if state_value < split[1]:
                # 0 is left
                return 0
            else:
                # 1 is right
                return 1
        else:
            if len(state) == 1:
                point = state
            else:
                axis = state[0]
                point = state[1]

                if axis != split[0]:
                    state = 0
                else:
                    state = point
            
            if point < split[1]:
                return 0
            else:
                return 1

    def get_current_policy(self):
        res = []
        for Node in self.nodes:
            if self.is_leaf[Node.node_id]:
                res.append(Node.val[0])

        return res

    def get_all_leaves(self):
        res = []
        for Node in self.nodes:
            if self.is_leaf[Node.node_id]:
                res.append(Node)

        return res
    
    def get_region(self, node, dim_state):
        region = np.zeros((dim_state, 2))
        for i in range(dim_state):
            # reset node to selected one
            test_node = node

            is_left = False
            is_right = False
            is_root = False
            lb = -np.inf
            ub = np.inf
            
            if test_node.id_left is not None or test_node.id_right is not None:
                print("[TREE POLICY] You are requesting a region for a non leaf!")
                return None

            # case we are root
            if test_node.id_father is None:
                region[i] = [lb, ub]
                continue
            
            father_node = self.nodes[test_node.id_father]
            while not ((is_left and is_right) or is_root):
                if father_node.id_left == test_node.node_id and not is_left:
                    ub = father_node.val[1][1]
                    is_left = True
                elif father_node.id_right == test_node.node_id and not is_right:
                    lb = father_node.val[1][1]
                    is_right = True

                if father_node.id_father is None:
                    is_root = True
                else:
                    test_node = father_node
                    father_node = self.nodes[father_node.id_father]
            
            region[i] = [lb, ub]

        return region
    
    def get_lower_vertex(self, node, dim_state):
        vertex = np.zeros(dim_state)
        region = self.get_region(node, dim_state)

        for i in range(dim_state):
            vertex[i] = region[i][0]

        return vertex
    
    def get_upper_vertex(self, node, dim_state):
        vertex = np.zeros(dim_state)
        region = self.get_region(node, dim_state)

        for i in range(dim_state):
            vertex[i] = region[i][1]

        return vertex

    def check_already_existing_split(self, value):
        for Node in self.nodes:
            if not self.is_leaf[Node.node_id] and Node.val is not None:
                if Node.val[1] == value:
                    return False
        return True

    def _to_dot(self, node, dot):
        if node:
            dot.node(str(node.node_id), label=str(node.val))
            if node.id_left is not None:
                dot.edge(str(node.node_id), str(self.nodes[node.id_left].node_id))
                self._to_dot(self.nodes[node.id_left], dot)
            if node.id_right is not None:
                dot.edge(str(node.node_id), str(self.nodes[node.id_right].node_id))
                self._to_dot(self.nodes[node.id_right], dot)

    def to_png(self, filename='binary_tree.png'):
        dot = graphviz.Digraph(format='png')
        self._to_dot(self.nodes[0], dot)
        dot.render(filename, format='png', cleanup=True)

    def to_list(self, node) -> None:
        if node is None:
            print('[TREE POLICY] Node is None')
            return
        
        print(f'Node information:\n Node id: {node.node_id}\n Parameter: {node.val[0]}\n Split point: {node.val[1]}\n Is leaf: {self.is_leaf[node.node_id]}\n')
    
