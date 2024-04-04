import numpy as np
import graphviz

class Node:
    def __init__(self, key, node_id, father=None):
        self.left = None
        self.right = None
        self.val = key
        self.node_id = node_id
        self.father = father


class BinaryTree:
    def __init__(self):
        self.root = None

        # todo fix use dict
        self.nodes = []

    def insert(self, key, split=None):
        node_id = len(self.nodes) + 1
        if self.root is None:
            self.root = Node([key, None], node_id)
            self.nodes.append(self.root)
        else:
            self._insert_recursive_father(key, self.root, node_id, split)

    def _insert_recursive_father(self, key, root, node_id, split):
        if root.val[1] is not None:
            if split < root.val[1]:
                self._insert_recursive_father(key, root.left, node_id, split)
            else:
                self._insert_recursive_father(key, root.right, node_id, split)
        else:
            self._insert_recursive_leaf(key, root, node_id, split)

    def _insert_recursive_leaf(self, key, root, node_id, split):
        # add split point
        root.val = [split if v is None else v for v in root.val]

        root.left = Node([key[0], None], node_id, root)
        root.right = Node([key[1], None], node_id + 1, root)
        self.nodes.append(root.left)
        self.nodes.append(root.right)

    def inorder_traversal(self, root):
        result = []
        if root:
            result += self.inorder_traversal(root.left)
            result.append((root.val, root.node_id))
            result += self.inorder_traversal(root.right)
        return result

    def search(self, key):
        return self._search_recursive(key, self.root) if self.root else False

    def _search_recursive(self, key, root):
        if root is None or root.val[0] == key:
            return root is not None

        if key < root.val[0]:
            return self._search_recursive(key, root.left)
        else:
            return self._search_recursive(key, root.right)

    def search_by_id(self, node_id):
        return self._search_by_id_recursive(node_id, self.root) if self.root else None

    def _search_by_id_recursive(self, node_id, root):
        if root is None or root.node_id == node_id:
            return root

        left_result = self._search_by_id_recursive(node_id, root.left)
        if left_result:
            return left_result

        return self._search_by_id_recursive(node_id, root.right)

    def find_closest_leaf(self, key):
        return self._find_closest_leaf(self.root, key)

    def _find_closest_leaf(self, root, key):
        if root.val[1] is None:
            return root

        if key < root.val[1]:
            # left_result = self._find_closest_leaf(root.left, key)
            # return left_result if left_result is not None else root
            return self._find_closest_leaf(root.left, key)
        else:
            return self._find_closest_leaf(root.right, key)

    # def find_associated_node(self, number):
    #     return self._find_associated_node(self.root, number)
    #
    # def _find_associated_node(self, root, number):
    #     if root is None:
    #         return None
    #
    #     if root.val is None or number < root.val[1]:
    #         return self._find_associated_node(root.left, number) or root
    #     else:
    #         return self._find_associated_node(root.right, number) or root

    def get_all_leaves(self):
        res = []
        for Node in self.nodes:
            if Node.val[1] is None:
                res.append(Node)

        return res

    def get_new_policy(self):
        res = []
        for Node in self.nodes:
            if Node.val[1] is None:
                res.append(Node.val[0].item())

        return res

    def get_father(self, node):
        leaves = self.get_all_leaves()
        res = None
        for order, Node in enumerate(leaves):
            if order == node:
                res = Node.father
                break

        return res

    def print_inorder(self):
        print("Inorder Traversal:", self.inorder_traversal(self.root))

    def print_all_nodes(self):
        print("All Nodes:")
        for node in self.nodes:
            print(f"Node ID: {node.node_id}, Value: {node.val}")

    def print_tree(self, node=None, level=0, prefix="Root: "):
        if node is None:
            node = self.root

        stack = [(node, level, prefix)]
        while stack:
            node, level, prefix = stack.pop()
            if node:
                print(" " * (level * 5) + prefix + f"|- {node.val} ({node.node_id})")
                stack.append((node.right, level + 1, " " * (level * 6) + "|- R: "))
                stack.append((node.left, level + 1, " " * (level * 6) + "|- L: "))

    def _to_dot(self, node, dot):
        if node:
            dot.node(str(node.node_id), label=str(node.val))
            if node.left:
                dot.edge(str(node.node_id), str(node.left.node_id))
                self._to_dot(node.left, dot)
            if node.right:
                dot.edge(str(node.node_id), str(node.right.node_id))
                self._to_dot(node.right, dot)

    def to_png(self, filename='binary_tree.png'):
        dot = graphviz.Digraph(format='png')
        self._to_dot(self.root, dot)
        dot.render(filename, format='png', cleanup=True)

# Example usage:
if __name__ == "__main__":
    tree = BinaryTree()
    keys = [5, 2, 3, 4, 7]

    # tree.insert(keys[0])
    # tree.insert(keys[1], 12)
    # tree.insert(keys[2], 12)
    #
    # tree.print_tree()

    for key in keys:
        tree.insert(key, key + 1)
    tree.print_tree()

    # tree.print_inorder()
    # tree.print_all_nodes()

    # Example search
    search_key = 4
    if tree.search(search_key):
        print(f"{search_key} is found in the tree.")
    else:
        print(f"{search_key} is not found in the tree.")

    associated_node_3 = tree.find_associated_node(4)
    associated_node_0_9 = tree.find_associated_node(8)

    print(f"\nNode associated with 3: {associated_node_3.val[0]}")
    print(f"Node associated with 0.9: {associated_node_0_9.val[0]}")

    tree.print_tree()
