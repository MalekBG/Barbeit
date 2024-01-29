from simulator import Simulator
from graphviz import Digraph
from collections import deque


class ScenarioNode:
    def __init__(self, assigned_tasks=None, available_resources=None, parent=None):
        self.assigned_tasks = assigned_tasks if assigned_tasks is not None else []
        self.available_resources = available_resources if available_resources is not None else []
        self.children = []
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)


class ScenarioTree:
    def __init__(self, max_depth=2):
        self.root = ScenarioNode()
        self.max_depth = max_depth

    def generate_tree(self, root, unassigned_tasks, resource_pool, max_depth=2):
        queue = deque([(root, unassigned_tasks, 0)])  # Queue contains tuples of (node, remaining_tasks, depth)

        while queue:
            current_node, tasks, depth = queue.popleft()

            if depth >= max_depth or not tasks:
                continue

            # Generate all possible child nodes for the current node
            for task in tasks:
                for resource in current_node.available_resources:
                    if resource in resource_pool[task.task_type]:
                        # Create a new child node with this task assigned to this resource
                        assigned_tasks = current_node.assigned_tasks + [(task, resource)]
                        available_resources = list(current_node.available_resources)
                        available_resources.remove(resource)
                        child_node = ScenarioNode(assigned_tasks, available_resources, current_node)
                        current_node.add_child(child_node)

                        # Prepare the remaining tasks for the child node
                        remaining_tasks = list(tasks)
                        remaining_tasks.remove(task)

                        # Add the child node to the queue with its remaining tasks and increased depth
                        queue.append((child_node, remaining_tasks, depth + 1))


class MyPlanner:
    def plan(self, scenario_tree, available_resources, unassigned_tasks, resource_pool):
        
        scenario_tree.root.available_resources = available_resources

        
        # Generate the scenario tree
        scenario_tree.generate_tree(scenario_tree.root, unassigned_tasks, resource_pool, scenario_tree.max_depth)


        if scenario_tree.root.children:
            return scenario_tree.root.assigned_tasks
        return []


    def report(self, event):
        print(event)
        

def visualize_scenario_tree(root):
    dot = Digraph(comment='Scenario Tree')
    dot.attr(rankdir='LR')  # Set graph orientation from left to right
    node_counter = 0  # Counter to keep track of node IDs

    def add_nodes_edges(node, parent_id=None):
        nonlocal node_counter
        node_id = str(node_counter)
        node_counter += 1

        # Create a label for the node based on assigned tasks
        label = ', '.join([f'T{t.id}-R{r}' for t, r in node.assigned_tasks]) if node.assigned_tasks else "Root"

        # Add the current node to the graph
        dot.node(node_id, label=label)

        # Connect the current node to its parent
        if parent_id is not None:
            dot.edge(parent_id, node_id)

        # Recursively add child nodes and edges
        for child in node.children:
            add_nodes_edges(child, node_id)

    # Start the recursive process from the root node
    add_nodes_edges(root)

    return dot

def print_tree_structure(node, depth=0):
    indent = "  " * depth  # Create indentation based on depth
    if node.assigned_tasks:
        tasks_str = ', '.join([f'T{t.id}-R{r}' for t, r in node.assigned_tasks])
    else:
        tasks_str = "Root"
    print(f"{indent}Node (Depth {depth}): {tasks_str}, Available Resources: {node.available_resources}")

    for child in node.children:
        print_tree_structure(child, depth + 1)



my_planner = MyPlanner()
scenario_tree = ScenarioTree(max_depth=2)
simulator = Simulator(my_planner, scenario_tree, "BPI Challenge 2017 - instance 2.pickle")
avg_cycle_time, completion_msg, scenario_tree = simulator.run(1)

# Visualize the complete scenario tree
dot = visualize_scenario_tree(scenario_tree.root)
dot.render('complete_scenario_tree', view=True, format='pdf')

# Print the simulation results
print(f"Average Cycle Time: {avg_cycle_time}")
print(completion_msg)

