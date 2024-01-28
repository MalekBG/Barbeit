from simulator import Simulator
from graphviz import Digraph


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

    def generate_tree(self, current_node, unassigned_tasks, resource_pool, depth=0):
        print(f"Generating tree at depth {depth} with {len(unassigned_tasks)} unassigned tasks")  # Debugging print
        
        if depth == self.max_depth or not unassigned_tasks:
            return

        for task in unassigned_tasks:
            for resource in current_node.available_resources:
                if resource in resource_pool[task.task_type]:
                    # Create a new child node with this task assigned to this resource
                    assigned_tasks = current_node.assigned_tasks + [(task, resource)]
                    available_resources = list(current_node.available_resources)
                    available_resources.remove(resource)
                    child_node = ScenarioNode(assigned_tasks, available_resources, current_node)
                    current_node.add_child(child_node)
                    print(f"Added child node with task {task.id} and resource {resource}")  # Debugging print

                    # Recursively generate the tree from this child node
                    remaining_tasks = list(unassigned_tasks)
                    remaining_tasks.remove(task)
                    self.generate_tree(child_node, remaining_tasks, resource_pool, depth + 1)


class MyPlanner:
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        # Initialize the scenario tree
        scenario_tree = ScenarioTree(max_depth=2)
        scenario_tree.root.available_resources = available_resources

        # Generate the scenario tree
        scenario_tree.generate_tree(scenario_tree.root, unassigned_tasks, resource_pool)

        # For simplicity, return the first scenario's assignments
        # In a real scenario, you would analyze the tree to choose the best scenario
        if scenario_tree.root.children:
            return scenario_tree.root.children[0].assigned_tasks
        return []


    def report(self, event):
        print(event)
        

def visualize_scenario_tree(root):
    dot = Digraph(comment='Scenario Tree')
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
avg_cycle_time, completion_msg, scenario_tree = simulator.run(1*5)
print_tree_structure(scenario_tree.root)

# Visualize the complete scenario tree
dot = visualize_scenario_tree(scenario_tree.root)
dot.render('complete_scenario_tree', view=True, format='pdf')

# Print the simulation results
print(f"Average Cycle Time: {avg_cycle_time}")
print(completion_msg)

