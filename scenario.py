import copy
import os
from simulator import Simulator
from graphviz import Digraph
from collections import deque
import threading
import time
import matplotlib.pyplot as plt
import psutil



class ScenarioNode:
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
        self.save_simulation_state(simulator.current_state, simulator.assigned_tasks, 'initial_state')
        
    def save_simulation_state(self, state, assignments, state_id):
            self.table[state_id] = (state, assignments)
         
    def load_simulation_state(self, state_id):
       #print("current table: ", self.table[state_id]) 
        state_to_load , assignments_to_load = copy.deepcopy(self.table[state_id])
        self.simulator.current_state = state_to_load
        for key, value in state_to_load.items():
            setattr(self.simulator, key, value)
        return assignments_to_load, state_to_load['now']



def explore_simulation(simulator, sim_state, scenario_tree, max_depth=4, bfs=True):
    # Initialize queues for state IDs and corresponding scenario tree nodes
    state_queue = deque(['Level 1'])
    node_queue = deque([scenario_tree.root])
    
    # Save the initial state and assignments
    state_id = 'Level 1'
    state, assignments = simulator.run()
    
    sim_state.save_simulation_state(state, assignments, state_id)
    
    while state_queue:
        current_state_id = state_queue.popleft()
        current_node = node_queue.popleft()
        current_depth = current_state_id.count('_')
        
        # Stop if the maximum depth is reached        
        if current_depth == max_depth:
            state_queue.appendleft(current_state_id)
            return
        
        else:
            # Load the current simulation state and assignments
            assignments, moment = sim_state.load_simulation_state(current_state_id)
                            
            for index, (task,resource) in enumerate(assignments):
                # Load the parent simulation state
                sim_state.load_simulation_state(current_state_id)
                
                # Construct new state IDs for children and for studying        
                child_state_id = current_state_id + '_' + f'{index+1}'
                state_to_study = current_state_id + '_' + f'child{index+1}'
                
                # Run the simulation for the current assignment        
                new_state, new_assignments = simulator.run(task, resource)
                
                # Save the new state and assignments
                sim_state.save_simulation_state(new_state, new_assignments, child_state_id)
                
                # Create a new child node and add it to the current node        
                child_node = ScenarioNode(task, resource, current_node, state_to_study, moment)
                current_node.add_child(child_node)
                
                # Add the new state ID and node to the queues
                if bfs:
                    # Use append for BFS
                    state_queue.append(child_state_id)
                    node_queue.append(child_node)
                else:
                    # Use appendleft for DFS
                    state_queue.appendleft(child_state_id)
                    node_queue.appendleft(child_node)



# Global lists to store usage data
memory_usage = []
cpu_usage = []
timestamps = []

#Monitors CPU and memory usage over time.
def monitor_resources(interval=0.1):
    global memory_usage, cpu_usage, timestamps
    memory_usage.clear()
    cpu_usage.clear()
    timestamps.clear()
    start_time = time.time()
    while monitoring:
        memory_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))  # Convert bytes to MB
        cpu_usage.append(psutil.cpu_percent())
        timestamps.append(time.time() - start_time)
        time.sleep(interval)
        
        
        
#Plots the CPU and memory usage over time.
def plot_resources():
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(timestamps, memory_usage, label='Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='red')
    plt.title('CPU Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()







my_planner = MyPlanner()
scenario_tree = ScenarioTree()
simulator = Simulator(my_planner, "BPI Challenge 2017 - instance 2.pickle")
sim_state = SimState(simulator)


# Flag to control the monitoring thread
monitoring = True

# Start the monitoring thread
monitor_thread = threading.Thread(target=monitor_resources, args=(0.01,))
monitor_thread.start()

explore_simulation(simulator, sim_state, scenario_tree, 5, bfs=True)

# Stop the monitoring thread
monitoring = False
monitor_thread.join()

# Debugging prints
#print(f"Data points collected: {len(memory_usage)}")
#print(f"Memory usage data: {memory_usage}")
#print(f"CPU usage data: {cpu_usage}")


print("Mean Timestamp: ", sum(timestamps)/len(timestamps))
print("Max Timestamp: ", max(timestamps))
print("Mean CPU usage: ", sum(cpu_usage)/len(cpu_usage))
print("Max CPU usage: ", max(cpu_usage)) 
print("Mean memory usage: ", sum(memory_usage)/len(memory_usage))
print("Max memory usage: ", max(memory_usage))
     

# Plot the resource usage
#plot_resources()


# Visualize the complete scenario tree
dot = scenario_tree.visualize_scenario_tree(scenario_tree.root)
dot.render('scenario_tree', view=True, format='pdf')

