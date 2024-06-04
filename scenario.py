import copy
import pandas as pd
from simulator import Simulator
from graphviz import Digraph
from collections import deque, defaultdict
from datetime import datetime
from numpy import float64
import threading
import psutil
import time



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
        node_counter = [0]  # Use a list for the counter

        def add_nodes_edges(node, parent_id=None):
            node_id = f'node{node_counter[0]}'
            node_counter[0] += 1

            # Special label for the root node
            if not node.parent:
                node_label = "Root"
            else:
                # Label for non-root nodes: shows ID and timestamp
                node_label = f"Assignment Update at: {node.timestamp}"

            # Add the current node to the graph
            dot.node(node_id, label=node_label)

            # Add an edge from the parent to the current node, if this is not the root node
            if parent_id is not None and node.assigned_task:
                # Construct the label for the edge using task-resource pairs
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
            # Check each available resource for suitability
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    # If valid, add the task-resource pair directly to the valid_assignments list
                    valid_assignments.append((task, resource))
        return valid_assignments



class SimState:
    def __init__(self, simulator):
        self.simulator = simulator
        self.table = {}
        self.leaf_states = []
        self.save_simulation_state(simulator.current_state, simulator.assigned_tasks, 'depth 0')
        
    def save_simulation_state(self, state, assignments, state_id):
            self.table[state_id] = (state, assignments)
         
    def load_simulation_state(self, state_id):
        state_to_load , assignments_to_load = copy.deepcopy(self.table[state_id])
        self.simulator.current_state = state_to_load
        for key, value in state_to_load.items():
            setattr(self.simulator, key, value)
        return assignments_to_load, state_to_load['now']



def explore_simulation(sim_state, max_depth=4):
    # Initialize queue for state IDs and depths
    state_queue = deque([(0, 'Level 1')])

    # Save the initial state and assignments
    state_id = 'Level 1'
    state, assignments = sim_state.simulator.run()
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
            new_state, new_assignments = sim_state.simulator.run(task, resource)

            # Save the new state and assignments
            sim_state.save_simulation_state(new_state, new_assignments, child_state_id)
            
            if current_depth == max_depth - 1:
                sim_state.leaf_states.append(child_state_id)

            # Add the new state ID and depth to the queue
            state_queue.append((current_depth + 1, child_state_id))
            
            
    print(f"Exploration finished with {len(sim_state.table)-1} total states.")
            


def build_scenario_tree(sim_state, scenario_tree, bfs=True):
    # Initialize queues for state IDs, corresponding scenario tree nodes, and depths
    state_queue = deque([(0, 'Level 1')])
    node_queue = deque([scenario_tree.root])
    
    node_counter = 1  # Start with 1 to count the root node

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
            
            # Increment the node counter
            node_counter += 1

            # Add the new state ID, node, and depth to the queues
            if bfs:
                state_queue.append((current_depth + 1, child_state_id))
                node_queue.append(child_node)
            else:
                state_queue.appendleft((current_depth + 1, child_state_id))
                node_queue.appendleft(child_node)
                
    # Print the total number of nodes in the built tree
    print(f"Built tree with {node_counter} total nodes.")
             


def explore_simulation_timed(sim_state, goal_timestamp=1):
    # Initialize queue for state IDs and timestamps
    state_queue = deque([('Level 1', 0)])  # Starting from the initial state with timestamp 0

    # Save the initial state and assignments
    state_id = 'Level 1'
    state, assignments = sim_state.simulator.run()
    sim_state.save_simulation_state(state, assignments, state_id)

    while state_queue:
        current_state_id, current_timestamp = state_queue.popleft()

        # Load the current simulation state and assignments
        assignments, current_timestamp = sim_state.load_simulation_state(current_state_id)

        # If there are no further assignments, the current node is a leaf node
        if not assignments or current_timestamp >= goal_timestamp:
            sim_state.leaf_states.append(current_state_id)
            continue

        # Process the assignments
        for index, (task, resource) in enumerate(assignments):
            sim_state.load_simulation_state(current_state_id)
            child_state_id = f'{current_state_id}_{index+1}'
            new_state, new_assignments = sim_state.simulator.run(task, resource)

            # Save the state and add it to the queue for further exploration
            sim_state.save_simulation_state(new_state, new_assignments, child_state_id)
            state_queue.append((child_state_id, new_state['now']))

            # If the new state has reached the goal timestamp or has no further assignments, it's a leaf node
            if new_state['now'] >= goal_timestamp or not new_assignments:
                sim_state.leaf_states.append(child_state_id)

    print(f"Exploration finished with {len(sim_state.table)-1} total states.")



def find_best_scenario(sim_state):
    print(f"Finding the best scenario from {len(sim_state.leaf_states)} generated scenarios.")
    min_avg = float('inf')  # Initialize min_avg to infinity
    best_scenario_id = None

    # Scroll through all state IDs in sim_state.leaf_states
    for state_id in sim_state.leaf_states:
        sim_state.load_simulation_state(state_id)  # Load the simulation state

        # Calculate the average of completed tasks per hour for the current state
        current_avg = sim_state.simulator.average_completed_tasks

        # Check if the current average is less than the current min_avg
        if current_avg < min_avg:
            min_avg = current_avg  # Update min_avg
            best_scenario_id = state_id  # Update best_scenario_id

    # Load the best scenario state
    sim_state.load_simulation_state(best_scenario_id)

    # Return the message with details of the best scenario
    print(f"Best generated scenario:\n{sim_state.simulator.allocated_tasks}\n" f"{min_avg:.2f} completed tasks per hour.\n"
            f"{sim_state.simulator.completed_tasks} total completed tasks.\n")



def calculate_distance(state1, state2):
    # Constants and Parameters
    rush_hour_start = 11  # Rush hour starts at 11 AM
    rush_hour_end = 13    # Rush hour ends at 1 PM

    # Helper function to calculate closeness to rush hour
    def closeness_to_rush_hour(timestamp):
        hour = datetime.fromtimestamp(timestamp).hour
        if hour < rush_hour_start:
            return rush_hour_start - hour
        elif hour > rush_hour_end:
            return hour - rush_hour_end
        else:
            return 0  # within rush hours

    # Modified weighted time difference calculation
    closeness1 = state1['now'] / (1 + closeness_to_rush_hour(state1['now']))
    closeness2 = state2['now'] / (1 + closeness_to_rush_hour(state2['now']))
    time_diff = abs(closeness1 - closeness2)

    # Event Queue Processing using the first event's timestamp
    next_event_time1 = state1['events'].pop(0)[0]
    next_event_time2 = state2['events'].pop(0)[0]
    event_time_diff = abs(next_event_time1 - next_event_time2)

    # Tasks and Resources
    unassigned_tasks_diff = abs((len(state1['unassigned_tasks']) / max(1, len(state1['unassigned_tasks']) + len(state1['assigned_tasks']))) -
                                    (len(state2['unassigned_tasks']) / max(1, len(state2['unassigned_tasks']) + len(state2['assigned_tasks']))))
    
    assigned_tasks_diff = abs((len(state1['assigned_tasks']) / max(1, len(state1['unassigned_tasks']) + len(state1['assigned_tasks']))) -
                                    (len(state2['assigned_tasks']) / max(1, len(state2['unassigned_tasks']) + len(state2['assigned_tasks']))))
    
    busy_cases_diff = abs((len(state1['busy_cases']) / max(1, len(state1['busy_cases']) + state1['finalized_cases'])) -
                                    (len(state2['busy_cases']) / max(1, len(state2['busy_cases']) + state2['finalized_cases'])))
    
    finalized_cases_diff = abs((state1['finalized_cases'] / max(1, len(state1['busy_cases']) + state1['finalized_cases'])) -
                                    (state2['finalized_cases'] / max(1, len(state2['busy_cases']) + state2['finalized_cases'])))   
    
    resource_utilization_diff = abs((len(state1['busy_resources']) / max(1, len(state1['available_resources']) + len(state1['busy_resources']) + len(state1['away_resources']) + len(state1['reserved_resources']))) -
                                    (len(state2['busy_resources']) / max(1, len(state2['available_resources']) + len(state2['busy_resources']) + len(state2['away_resources']) + len(state2['reserved_resources']))))
    
    resource_reserved_diff = abs((len(state1['reserved_resources']) / max(1, len(state1['available_resources']) + len(state1['busy_resources']) + len(state1['away_resources']) + len(state1['reserved_resources']))) -
                                    (len(state2['reserved_resources']) / max(1, len(state2['available_resources']) + len(state2['busy_resources']) + len(state2['away_resources']) + len(state2['reserved_resources']))))
    # Simple attribute comparisons
    simple_attributes = ['total_cycle_time', 'completed_tasks', 'average_completed_tasks']
    
    simple_diffs = sum(abs(len(state1[attr]) - len(state2[attr])) if not (isinstance(state1[attr], int) or isinstance(state1[attr], float)) else abs(state1[attr] - state2[attr])
                       for attr in simple_attributes)
    
    different_elements_counter = 0
    total_length_weight = 0
    
    # Attributes to compare using the count_unique_elements function
    attributes_to_compare = [
        'unassigned_tasks', 'assigned_tasks', 'available_resources',
        'busy_resources', 'reserved_resources', 'away_resources',
        'busy_cases'
    ]

    # Iterate over each attribute and calculate the differences
    for attr in attributes_to_compare:
        attr1 = state1[attr]
        attr2 = state2[attr]
        total_length_weight += len(attr1) + len(attr2)
        if attr1 is not None and attr2 is not None:
            different_elements_counter += count_unique_elements(attr1, attr2)
                  
            
    different_elements_weighted = different_elements_counter / total_length_weight

    # Combine all metrics into a single distance measure
    distance = (time_diff + event_time_diff + assigned_tasks_diff + unassigned_tasks_diff + busy_cases_diff + finalized_cases_diff +
                resource_utilization_diff + resource_reserved_diff + simple_diffs) * (1 + different_elements_weighted)
    return distance



def count_unique_elements(attr1, attr2):
    if isinstance(attr1, set) and isinstance(attr2, set):
        unique_items = attr1.symmetric_difference(attr2)
    elif isinstance(attr1, dict) and isinstance(attr2, dict):
        unique_items = set(attr1.keys()).symmetric_difference(set(attr2.keys()))
    elif isinstance(attr1, list) and isinstance(attr2, list):
        unique_items = set(attr1).symmetric_difference(set(attr2))
    else:
        # Fallback for unsupported types
        print("Unsupported types for comparison.")
        return 0
    return len(unique_items)



def explore_and_merge_simulation_fct(sim_state, max_depth=4, merge_threshold=0.5):
    # Initialize queue for state IDs and depths
    state_queue = deque([(0, 'Level 1')])

    # Save the initial state and assignments
    state_id = 'Level 1'
    state, assignments = sim_state.simulator.run()
    sim_state.save_simulation_state(state, assignments, state_id)
    deleted_states = 0
    average_distance = 0

    # A list to store state IDs at each depth for potential merging
    depth_states = {i: [] for i in range(max_depth + 1)}

    while state_queue:
        current_depth, current_state_id = state_queue.popleft()
        depth_states[current_depth].append(current_state_id)
        
        if current_state_id not in sim_state.table:
            continue

        # Stop if the maximum depth is reached
        if current_depth == max_depth:
            continue

        # Load the current simulation state and assignments
        assignments, _ = sim_state.load_simulation_state(current_state_id)

        for index, (task, resource) in enumerate(assignments):
            print(f"Exploring state {current_state_id} at depth {current_depth} with assignment {index+1}")
            sim_state.load_simulation_state(current_state_id)
            
            # Construct new state IDs for children
            child_state_id = f'{current_state_id}_{index+1}'

            # Run the simulation for the current assignment
            new_state, new_assignments = sim_state.simulator.run(task, resource)

            # Save the new state and assignments
            sim_state.save_simulation_state(new_state, new_assignments, child_state_id)
            
            if current_depth == max_depth - 1:
                sim_state.leaf_states.append(child_state_id)

            # Add the new state ID and depth to the queue
            state_queue.append((current_depth + 1, child_state_id))
        
        # Starting from depth 7, merge states within the same depth if close enough
        if current_depth >= 6:
            if state_queue:
                next_element_depth, next_element_state_id = state_queue.popleft()
                if next_element_depth == current_depth + 1:
                    state_queue.appendleft((next_element_depth, next_element_state_id))
                    to_merge = []
                    states_to_compare = depth_states[current_depth]
                    distances = []
                    while states_to_compare:
                        state_id1 = states_to_compare.pop()
                        if state_id1 in to_merge:
                            continue
                        for state_id2 in states_to_compare:
                            print(f"Comparing states {state_id1} and {state_id2} at depth {current_depth}")
                         # Load and copy the first state
                            sim_state.load_simulation_state(state_id1)
                            state1 = copy.deepcopy(sim_state.simulator.current_state)
                    
                            # Load and copy the second state
                            sim_state.load_simulation_state(state_id2)
                            state2 = copy.deepcopy(sim_state.simulator.current_state)
                    
                            distance = calculate_distance(state1, state2)
                            distances.append(distance)
                    
                            if distance <= merge_threshold:
                                # Compare average completed tasks and keep the state with the higher average
                                if state1['average_completed_tasks'] > state2['average_completed_tasks']:
                                    to_merge.append((state_id2, state_id1))  # Remove state_id2, keep state_id1
                                else:
                                    to_merge.append((state_id1, state_id2))  # Remove state_id1, keep state_id2
                    if len(distances) > 0:
                        average_distance = sum(distances) / len(distances)                
                        print(f"Average distance between states at depth {current_depth}: {distance:.2f}")
                    # Perform the merging by removing states and updating the remaining states in depth_states
                    for remove_id, keep_id in to_merge:
                        if (remove_id in sim_state.table):    
                            del sim_state.table[remove_id]
                            deleted_states += 1
                        print(f"Removed state {remove_id} and kept state {keep_id} at depth {current_depth}")
                else:
                    state_queue.appendleft((next_element_depth, next_element_state_id))
                    

    print(f"Exploration finished with {deleted_states} states removed and {len(sim_state.table)-1} total states remaining.")
    return average_distance, deleted_states



# Monitors memory usage over time.
def monitor_resources(interval=0.1):
    global memory_usage
    memory_usage= []
    while monitoring:
        memory_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))  # Convert bytes to MB
        time.sleep(interval)





# Initialize lists to store data for each run
state_numbers = []
average_memory_usages = []
peak_memory_usages = []
states_deleted = []
distance_states = []

# Global list to store memory usage data
memory_usage = []


# Run the code snippet 50 times
for i in range(50):
    print(f"Run {i+1}")
    # Reset the monitoring flag and start time
    monitoring = True

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(0.01,))
    monitor_thread.start()
    
    
    my_planner = MyPlanner()
    simulator = Simulator(my_planner, "BPI Challenge 2017 - instance 2.pickle")
    sim_state = SimState(simulator)
    #explore_simulation_timed(sim_state, goal_timestamp=0.2)
    distance, deleted = explore_and_merge_simulation_fct(sim_state, max_depth=7, merge_threshold=10)
    state_numbers.append(len(sim_state.table)-1)
    states_deleted.append(deleted)
    distance_states.append(distance)
    
    monitoring = False
    monitor_thread.join()
    

    # Calculate and store the required data
    average_memory_usages.append(sum(memory_usage) / len(memory_usage))
    peak_memory_usages.append(max(memory_usage))
    time.sleep(10)



# Calculate averages and peaks from the collected data
average_state_number = sum(state_numbers) / len(state_numbers)
peak_state_number = max(state_numbers)

# Create a DataFrame to store the collected data using lists
df1 = pd.DataFrame({
    'Average State Number': [average_state_number],
    'Peak State Number': [peak_state_number],
    'Average States Deleted': [sum(states_deleted) / len(states_deleted)],
    'Peak States Deleted': [max(states_deleted)],
    'Average Distance': [sum(distance_states) / len(distance_states)],
    'Peak Distance': [max(distance_states)]
})

df2 = pd.DataFrame({
    'Average Memory Usage (in MB)': average_memory_usages,
    'Peak Memory Usage (in MB)': peak_memory_usages
})

# Save the DataFrame to an Excel file
df1.to_excel('state_data.xlsx', index=False)
df2.to_excel('memory_data.xlsx', index=False)

'''
scenario_tree= ScenarioTree()
build_scenario_tree(sim_state, scenario_tree, True)

# Visualize the complete scenario tree
dot = scenario_tree.visualize_scenario_tree(scenario_tree.root)
dot.render('scenario_tree', view=True, format='pdf')'''
 