import copy
import pandas as pd
from simulator import Simulator
from graphviz import Digraph
from collections import deque
import threading
import time
import matplotlib.pyplot as plt
import psutil
from memory_profiler import profile


class ScenarioNode:
    __slots__ = ['assigned_task', 'assigned_resource', 'children', 'parent', 'state_id', 'timestamp']

    def __init__(self, assigned_task=None, assigned_resource=None, parent=None, state_id=None, timestamp=None):
        self.assigned_task = assigned_task if assigned_task is not None else []
        self.assigned_resource = assigned_resource if assigned_resource is not None else []
        self.children = []
        self.parent = parent
        self.state_id = state_id
        self.timestamp = timestamp

    def add_child(self, child):
        self.children.append(child)
                            


class ScenarioTree:
    def __init__(self):
        self.root = ScenarioNode()
        self.current_node = self.root
        self.root.state_id = 'initial_state'
        
    def visualize_scenario_tree(self, root):
        dot = Digraph(comment='Scenario Tree')
        dot.attr(rankdir='LR')  # Set graph orientation from left to right
        node_counter = [0]  # Use a list for the counter to avoid issues with closures in nested functions

        def add_nodes_edges(node, parent_id=None):
            node_id = f'node{node_counter[0]}'
            node_counter[0] += 1

            # Special label for the root node
            if not node.parent:
                node_label = "Root"
            else:
                # Label for non-root nodes: shows ID and timestamp
                node_label = f"ID:{node.state_id}, T:{node.timestamp}"

            # Add the current node to the graph
            dot.node(node_id, label=node_label)

            # Add an edge from the parent to the current node, if this is not the root node
            if parent_id is not None and node.assigned_task:
                # Construct the label for the edge using task-resource pairs
                # This concatenates all task-resource pairs into the edge label
                edge_label = ', '.join([f'T{node.assigned_task.id}-R{node.assigned_resource}'])
                dot.edge(parent_id, node_id, label=edge_label)
            # Recursively add child nodes and edges
            for child in node.children:
                add_nodes_edges(child, node_id)

        # Initialize the recursive process starting from the root node
        add_nodes_edges(root)

        return dot



class MyPlanner:
    def report(self, event):
        print(event)
        
    def plan(self, available_resources, unassigned_tasks, resource_pool):
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



class SimState:
    def __init__(self, simulator):
        self.simulator = simulator
        self.table = {}
        self.save_simulation_state(simulator.current_state, simulator.assigned_tasks, 'depth 0')
        
    def save_simulation_state(self, state, assignments, state_id):
            self.table[state_id] = (state, assignments)
         
    def load_simulation_state(self, state_id):
       #print("current table: ", self.table[state_id]) 
        state_to_load , assignments_to_load = copy.deepcopy(self.table[state_id])
        self.simulator.current_state = state_to_load
        for key, value in state_to_load.items():
            setattr(self.simulator, key, value)
        return assignments_to_load, state_to_load['now']




def explore_simulation(simulator, sim_state, max_depth=4):
    # Initialize queue for state IDs and depths
    state_queue = deque([(0, 'Level 1')])

    # Save the initial state and assignments
    state_id = 'Level 1'
    state, assignments = simulator.run()
    sim_state.save_simulation_state(state, assignments, state_id)

    while state_queue:
        current_depth, current_state_id = state_queue.popleft()

        # Stop if the maximum depth is reached
        if current_depth == max_depth:
            continue

        # Load the current simulation state and assignments
        assignments, _ = sim_state.load_simulation_state(current_state_id)

        for index, (task, resource) in enumerate(assignments):
            sim_state.load_simulation_state(current_state_id)
            
            # Construct new state IDs for children
            child_state_id = f'{current_state_id}_{index+1}'

            # Run the simulation for the current assignment
            new_state, new_assignments = simulator.run(task, resource)

            # Save the new state and assignments
            sim_state.save_simulation_state(new_state, new_assignments, child_state_id)

            # Add the new state ID and depth to the queue
            state_queue.append((current_depth + 1, child_state_id))
            


def build_scenario_tree(sim_state, scenario_tree, bfs=True):
    # Initialize queues for state IDs, corresponding scenario tree nodes, and depths
    state_queue = deque([(0, 'Level 1')])
    node_queue = deque([scenario_tree.root])

    while state_queue:
        current_depth, current_state_id = state_queue.popleft()
        current_node = node_queue.popleft()

        # Load the simulation state and assignments for the current node
        assignments, moment = sim_state.load_simulation_state(current_state_id)

        for index, (task, resource) in enumerate(assignments):
            # Construct state ID for child nodes
            child_state_id = f'{current_state_id}_{index+1}'

            # Check if child state exists in sim_state, continue if not
            if child_state_id not in sim_state.table:
                continue

            # Create a new child node
            child_node = ScenarioNode(task, resource, current_node, child_state_id, moment)
            current_node.add_child(child_node)

            # Add the new state ID, node, and depth to the queues
            if bfs:
                state_queue.append((current_depth + 1, child_state_id))
                node_queue.append(child_node)
            else:
                state_queue.appendleft((current_depth + 1, child_state_id))
                node_queue.appendleft(child_node)
                
                
                
# Global list to store memory usage data
#memory_usage = []

# Monitors memory usage over time.
def monitor_resources(interval=0.1):
    global memory_usage
    memory_usage = []
    start_time = time.time()
    while monitoring:
        memory_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))  # Convert bytes to MB
        time.sleep(interval)


# Plots the memory usage over time.
'''def plot_resources():
    plt.figure(figsize=(6, 6))
    plt.plot(timestamps, memory_usage, label='Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.show()'''



# Initialize lists to store data for each run
#average_memory_usages = []
#peak_memory_usages = []
#execution_times = []

# Run the code snippet 10 times
for i in range(1):
    # Reset the monitoring flag and start time
    #monitoring = True
    #start_time = time.time()

    # Start the monitoring thread
    #monitor_thread = threading.Thread(target=monitor_resources, args=(0.01,))
    #monitor_thread.start()

    
    my_planner = MyPlanner()
    simulator = Simulator(my_planner, "BPI Challenge 2017 - instance 2.pickle")
    sim_state = SimState(simulator)
    explore_simulation(simulator, sim_state, 9)

    # Stop the monitoring thread and calculate execution time
    #monitoring = False
    #monitor_thread.join()
    #end_time = time.time()

    # Calculate and store the required data
    #average_memory_usages.append(sum(memory_usage) / len(memory_usage))
    #peak_memory_usages.append(max(memory_usage))
    #execution_times.append(end_time - start_time)

# Create a DataFrame to store the collected data
#df = pd.DataFrame({
    #'Average Memory Usage (MB)': average_memory_usages,
    #'Peak Memory Usage (MB)': peak_memory_usages,
    #'Execution Time (s)': execution_times
#})

# Save the DataFrame to an Excel file
#df.to_excel('performance_data.xlsx', index=False)
     
     
#Plot the resource usage
#plot_resources()


scenario_tree= ScenarioTree()
build_scenario_tree(sim_state, scenario_tree, True)

# Visualize the complete scenario tree
dot = scenario_tree.visualize_scenario_tree(scenario_tree.root)
dot.render('scenario_tree', view=True, format='png')