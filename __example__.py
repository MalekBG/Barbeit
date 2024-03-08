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
    def __init__(self, sim_state = None, max_depth=2, current_node=None):
        self.root = ScenarioNode()
        self.sim_state = sim_state
        self.max_depth = max_depth
        self.current_node = current_node


    '''def generate_tree(self, root, unassigned_tasks, resource_pool, max_depth=2):
        
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
                        assigned_tasks = current_node.assigned_tasks.copy()
                        assigned_tasks.append((task, resource))
                        available_resources = set(current_node.available_resources)
                        available_resources.remove(resource)
                        child_id = ', '.join([f'T{t.id}-R{r}-TS:{self.sim_state.simulator.now}' for t, r in assigned_tasks]) if assigned_tasks else 'None'
                        child_node = ScenarioNode(assigned_tasks, available_resources, current_node, child_id, self.sim_state.simulator.now)
                        current_node.add_child(child_node)
                        #self.sim_state.save_simulation_state(f'{child_id}.pickle')

                        # Prepare the remaining tasks for the child node
                        remaining_tasks = tasks.copy()
                        remaining_tasks.remove(task)

                        # Add the child node to the queue with its remaining tasks and increased depth
                        queue.append((child_node, remaining_tasks, depth + 1))'''
                        
    
    def generate_child_node(self, current_node, assignment, available_resources, timestamp):
        # Create a copy of the current node's assigned tasks and add the new assignment
        new_assigned_tasks = current_node.assigned_tasks.copy()
        task, resource = assignment
        new_assigned_tasks.append((task, resource))

        # Update available resources and unassigned tasks based on the assignment
        new_available_resources = set(available_resources)
        #new_available_resources.remove(assignment[1])  # Remove the assigned resource
        '''if assignment[1] in new_available_resources:
            new_available_resources.remove(assignment[1])  # Safely remove the assigned resource
        else:
            print(f"Trying to remove a resource that's not available: {assignment[1]}")
            print(f"Current available resources: {new_available_resources}")'''

        # Create the new child node
        child_node = ScenarioNode(
            new_assigned_tasks,
            new_available_resources,
            current_node,
            ', '.join(f'T{task.id}-R{resource}-TS:{timestamp}') if assignment else 'None',
            timestamp
        )

        # Add the new child node to the current node
        current_node.add_child(child_node)

        return child_node



class MyPlanner:
    '''def plan(self, scenario_tree, available_resources, unassigned_tasks, resource_pool):
        scenario_tree.root.available_resources = available_resources
        
        # Generate the scenario tree
        scenario_tree.generate_tree(scenario_tree.root, unassigned_tasks, resource_pool, scenario_tree.max_depth)
        assignments = []
        
        if scenario_tree.root.children:
            for child in scenario_tree.root.children:
                if child.assigned_tasks:
                    assignments.extend(child.assigned_tasks)
            return assignments
        else:        
            return []'''


    def report(self, event):
        print(event)

        
    def plan_selected(self, available_resources, unassigned_tasks, resource_pool):
        valid_assignments = []

        # Iterate through each task
        for task in unassigned_tasks:
            task_valid_assignments = []

            # Check each available resource for suitability
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    # If valid, add the task-resource pair to the task-specific list
                    task_valid_assignments.append((task, resource))

            # If there are valid assignments for this task, add them to the overall list
            if task_valid_assignments:
                valid_assignments.extend(task_valid_assignments)

        # At this point, valid_assignments is a list of lists, where each sublist contains
        # all valid assignments for a specific task

        return valid_assignments
    
    

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
simulator = Simulator(my_planner, scenario_tree, "BPI Challenge 2017 - instance 2.pickle")
sim_state = SimState(simulator)
scenario_tree.sim_state = sim_state
# Initial message to the user
while True:
    user_input = input("Write 'auto' if you want the simulation to run automatically for the specified running time or 'stop' to stop at each decision point: ").lower()
    if user_input == 'auto':
        avg_cycle_time, completion_msg, scenario_tree = simulator.run(5, auto=True)
        break
    elif user_input == 'stop':
        avg_cycle_time, completion_msg, scenario_tree = simulator.run(5, auto=False)
        break
    else:
        print("Incorrect answer, try again:")
  


# Visualize the complete scenario tree
dot = visualize_scenario_tree(scenario_tree.root)
dot.render('scenario_tree', view=True, format='pdf')

# Print the simulation results
print(f"Average Cycle Time: {avg_cycle_time}")
print(completion_msg)