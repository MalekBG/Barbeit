from simulator import Simulator
from graphviz import Digraph
from collections import deque
from collections import defaultdict
import heapq
import pickle



class ScenarioNode:
    def __init__(self, assigned_tasks=None, available_resources=None, parent=None, state_id=None, timestamp=None):
        self.assigned_tasks = assigned_tasks if assigned_tasks is not None else []
        self.available_resources = available_resources if available_resources is not None else []
        self.children = []
        self.parent = parent
        self.state_id = state_id
        self.timestamp = timestamp


    def add_child(self, child):
        self.children.append(child)
                            


class ScenarioTree:
    def __init__(self, sim_state = None, max_depth=2):
        self.root = ScenarioNode()
        self.sim_state = sim_state
        self.max_depth = max_depth


    def generate_tree(self, root, unassigned_tasks, resource_pool, max_depth=2):
        #print("Generating tree with root node")
        #print("Root Available Resources:", root.available_resources)
       #print("Root Assigned Tasks:", root.assigned_tasks)
        queue = deque([(root, unassigned_tasks, 0)])  # Queue contains tuples of (node, remaining_tasks, depth)

        while queue:
            current_node, tasks, depth = queue.popleft()

            if depth >= max_depth or not tasks:
                #print('assigned_tasks after:', root.assigned_tasks)
                continue

            # Generate all possible child nodes for the current node
            #print(f"Processing node at depth {depth} with tasks:", tasks)
            for task in tasks:
                for resource in current_node.available_resources:
                    if resource in resource_pool[task.task_type]:
                        # Create a new child node with this task assigned to this resource
                        assigned_tasks = current_node.assigned_tasks.copy()
                        assigned_tasks.append((task, resource))
                        available_resources = set(current_node.available_resources)
                        available_resources.remove(resource)
                        child_id = ', '.join([f'T{t.id}-R{r}-TS:{self.sim_state.simulator.now}' for t, r in assigned_tasks]) if assigned_tasks else 'None'
                        child_node = ScenarioNode(assigned_tasks, available_resources, current_node, child_id, self.sim_state.simulator.now)
                        root.assigned_tasks.append((task, resource))
                        #print(f"Assigning Task {task.id} to Resource {resource}")
                        #print(f"Created Child Node ID: {child_id}")
                        current_node.add_child(child_node)
                        #self.sim_state.save_simulation_state(f'{child_id}.pickle')

                        # Prepare the remaining tasks for the child node
                        remaining_tasks = tasks.copy()
                        remaining_tasks.remove(task)

                        # Add the child node to the queue with its remaining tasks and increased depth
                        #print(f"Adding Child Node to Queue - ID: {child_id}, Remaining Tasks:", remaining_tasks)
                        queue.append((child_node, remaining_tasks, depth + 1))



class MyPlanner:
    def plan(self, scenario_tree, available_resources, unassigned_tasks, resource_pool):
        #print("Starting plan() method")
        #print("Unassigned Tasks:", unassigned_tasks)
        scenario_tree.root.available_resources = available_resources
        
        # Generate the scenario tree
        scenario_tree.generate_tree(scenario_tree.root, unassigned_tasks, resource_pool, scenario_tree.max_depth)
        #print('Children of root:', scenario_tree.root.children)
        
        if scenario_tree.root.children:
            return scenario_tree.root.assigned_tasks
        
        return []


    def report(self, event):
        print(event)
        
       
        
def remove_duplicates(root):
    if not root or not root.children:
        return

    queue = deque([(root, 0)])  # Queue contains tuples of (node, depth)
    current_depth_nodes = []  # To store nodes at the current depth
    last_depth = 0

    while queue:
        node, depth = queue.popleft()

        # Process nodes at the same depth
        if depth == last_depth:
            current_depth_nodes.append(node)
        else:
            # Process the previous depth nodes before moving to the next depth
            process_depth_nodes(current_depth_nodes)
            current_depth_nodes = [node]
            last_depth = depth

        # Add child nodes to the queue
        for child in node.children:
            queue.append((child, depth + 1))

    # Process the last set of nodes
    process_depth_nodes(current_depth_nodes)
    
    

def visualize_scenario_tree(root):
    dot = Digraph(comment='Scenario Tree')
    dot.attr(rankdir='LR')  # Set graph orientation from left to right
    node_counter = [0]  # Use a list for the counter to avoid issues with closures in nested functions

    def add_nodes_edges(node, parent_id=None):
        node_id = f'node{node_counter[0]}'
        node_counter[0] += 1

        # Label for the node: use timestamp, default to "Root" for the root node
        node_label = str(node.timestamp) if node.timestamp is not None else "Root"

        # Add the current node to the graph
        dot.node(node_id, label=node_label)

        # Add an edge from the parent to the current node, if this is not the root node
        if parent_id is not None and node.assigned_tasks:
            # Construct the label for the edge using task-resource pairs
            # This concatenates all task-resource pairs into the edge label
            edge_label = ', '.join([f'T{t.id}-R{r}' for t, r in node.assigned_tasks])
            dot.edge(parent_id, node_id, label=edge_label)
        # Recursively add child nodes and edges
        for child in node.children:
            add_nodes_edges(child, node_id)

    # Initialize the recursive process starting from the root node
    add_nodes_edges(root)

    return dot



def process_depth_nodes(nodes):
    if not nodes:
        return

    # Group nodes by their task-resource assignments
    assignment_groups = defaultdict(list)
    for node in nodes:
        key = ', '.join([f'T{t.id}-R{r}' for t, r in node.assigned_tasks])
        assignment_groups[key].append(node)

    # For each group, keep the node with the most children and remove the others
    for key, group_nodes in assignment_groups.items():
        if len(group_nodes) > 1:
            # Find the node with the most children
            max_children_node = heapq.nlargest(1, group_nodes, key=lambda n: len(n.children))[0]
            # Remove all other nodes from their parent's children list
            for node in group_nodes:
                if node is not max_children_node:
                    if node.parent:
                        node.parent.children.remove(node)



class SimState:
    def __init__(self, simulator):
        self.simulator = simulator
        with open('initial_state.pickle', 'wb') as file:
            pickle.dump(self.simulator.current_state, file)
        
        
    def save_simulation_state(self, filename):
        self.simulator.current_state = {
			'now': self.simulator.now,
			'events': self.simulator.events,
			'unassigned_tasks': self.simulator.unassigned_tasks,
			'assigned_tasks': self.simulator.assigned_tasks,
			'available_resources': self.simulator.available_resources,
			'busy_resources': self.simulator.busy_resources,
			'reserved_resources': self.simulator.reserved_resources,
			'busy_cases': self.simulator.busy_cases,
            'away_resources': self.simulator.away_resources,
            'away_resources_weights': self.simulator.away_resources_weights,
            'finalized_cases': self.simulator.finalized_cases,
            'total_cycle_time': self.simulator.total_cycle_time,
            'auto': self.simulator.auto
		}
        with open(filename, 'wb') as file:
            pickle.dump(self.simulator.current_state, file)
            
            
    def load_simulation_state(self, filename):
        with open(filename, 'rb') as file:
            self.simulator.current_state = pickle.load(file)
            self.simulator.now = self.simulator.current_state['now']
            self.simulator.events = self.simulator.current_state['events']
            self.simulator.unassigned_tasks = self.simulator.current_state['unassigned_tasks']
            self.simulator.assigned_tasks = self.simulator.current_state['assigned_tasks']
            self.simulator.available_resources = self.simulator.current_state['available_resources']
            self.simulator.busy_resources = self.simulator.current_state['busy_resources']
            self.simulator.reserved_resources = self.simulator.current_state['reserved_resources']
            self.simulator.busy_cases = self.simulator.current_state['busy_cases']
            self.simulator.away_resources = self.simulator.current_state['away_resources']
            self.simulator.away_resources_weights = self.simulator.current_state['away_resources_weights']
            self.simulator.finalized_cases = self.simulator.current_state['finalized_cases']
            self.simulator.total_cycle_time = self.simulator.current_state['total_cycle_time']
            self.simulator.auto = self.simulator.current_state['auto']
            
    
    









my_planner = MyPlanner()
scenario_tree = ScenarioTree(max_depth=3)
simulator = Simulator(my_planner, scenario_tree, "BPI Challenge 2017 - instance.pickle")
sim_state = SimState(simulator)
scenario_tree.sim_state = sim_state
#avg_cycle_time, completion_msg, scenario_tree = simulator.run(5)
#remove_duplicates(scenario_tree.root)

# Visualize the complete scenario tree
#dot = visualize_scenario_tree(scenario_tree.root)
#dot.render('scenario_tree', view=True, format='pdf')

# Print the simulation results
#print(f"Average Cycle Time: {avg_cycle_time}")
#print(completion_msg)
print(simulator.run(5))