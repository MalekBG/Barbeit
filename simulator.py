from collections import deque
from enum import Enum, auto
from datetime import datetime, timedelta
import random
from problems import MinedProblem
import pickle

RUNNING_TIME = 24 * 365


class Event:
	initial_time = datetime(2020, 1, 1)
	time_format = "%Y-%m-%d %H:%M:%S.%f"

	def __init__(self, case_id, task, timestamp, resource, lifecycle_state):
		self.case_id = case_id
		self.task = task
		self.timestamp = timestamp
		self.resource = resource
		self.lifecycle_state = lifecycle_state

	def __str__(self):
		t = (self.initial_time + timedelta(hours=self.timestamp)).strftime(self.time_format)
		if self.task is not None:
			tt = self.task.task_type
			dt = '\t' + '\t'.join(str(v) for v in self.task.data.values())
		else:
			tt = str(None)
			dt = ""
		return str(self.case_id) + "\t" + tt + "\t" + t + "\t" + str(self.resource) + "\t" + str(self.lifecycle_state) + dt


class EventType(Enum):
	CASE_ARRIVAL = auto()
	START_TASK = auto()
	COMPLETE_TASK = auto()
	PLAN_TASKS = auto()
	TASK_ACTIVATE = auto()
	TASK_PLANNED = auto()
	COMPLETE_CASE = auto()
	SCHEDULE_RESOURCES = auto()


class TimeUnit(Enum):
	SECONDS = auto()
	MINUTES = auto()
	HOURS = auto()
	DAYS = auto()


class SimulationEvent:
	def __init__(self, event_type, moment, task, resource=None, nr_tasks=0, nr_resources=0):
		self.event_type = event_type
		self.moment = moment
		self.task = task
		self.resource = resource
		self.nr_tasks = nr_tasks
		self.nr_resources = nr_resources

	def __lt__(self, other):
		return self.moment < other.moment

	def __str__(self):
		return str(self.event_type) + "\t(" + str(round(self.moment, 2)) + ")\t" + str(self.task) + "," + str(
			self.resource)


class Simulator:
	def __init__(self, planner, scenario_tree, instance_file="BPI Challenge 2017 - instance.pickle"):
		self.events = []
		self.unassigned_tasks = dict()
		self.assigned_tasks = dict()
		self.available_resources = set()
		self.away_resources = []
		self.away_resources_weights = []
		self.busy_resources = dict()
		self.busy_cases = dict()
		self.reserved_resources = dict()
		self.now = 0
		self.finalized_cases = 0
		self.total_cycle_time = 0
		self.case_start_times = dict()
		self.problem = MinedProblem.from_file(instance_file)
		self.problem.interarrival_time._alpha *= 4.8
		self.planner = planner
		self.scenario_tree = scenario_tree
		self.problem_resource_pool = self.problem.resource_pools
		self.current_state = None
		self.current_node = self.scenario_tree.root
		self.init_simulation()

	def init_simulation(self):
		for r in self.problem.resources:
			self.available_resources.add(r)
		self.events.append((0, SimulationEvent(EventType.SCHEDULE_RESOURCES, 0, None)))
		self.problem.restart()
		(t, task) = self.problem.next_case()
		self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))
		self.current_state = {
			'now': self.now,
			'events': self.events,
			'unassigned_tasks': self.unassigned_tasks,
			'assigned_tasks': self.assigned_tasks,
			'available_resources': self.available_resources,
			'busy_resources': self.busy_resources,
			'reserved_resources': self.reserved_resources,
			'busy_cases': self.busy_cases,
			'away_resources': self.away_resources,
			'away_resources_weights': self.away_resources_weights,
			'finalized_cases': self.finalized_cases,
			'total_cycle_time': self.total_cycle_time,
		}
		self.scenario_tree.root.state_id = 'initial_state'
		self.scenario_tree.current_node = self.scenario_tree.root

	def desired_nr_resources(self):
		return self.problem.schedule[int(self.now % len(self.problem.schedule))]

	def working_nr_resources(self):
		return len(self.available_resources) + len(self.busy_resources) + len(self.reserved_resources)

	def load_simulation_state(self, filename):
		with open(filename, 'rb') as file:
			state_to_load = pickle.load(file)
		self.now = state_to_load['now']
		self.events = state_to_load['events']
		self.unassigned_tasks = state_to_load['unassigned_tasks']
		self.assigned_tasks = state_to_load['assigned_tasks']
		self.available_resources = state_to_load['available_resources']
		self.busy_resources = state_to_load['busy_resources']
		self.reserved_resources = state_to_load['reserved_resources']
		self.busy_cases = state_to_load['busy_cases']
		self.away_resources = state_to_load['away_resources']
		self.away_resources_weights = state_to_load['away_resources_weights']
		self.finalized_cases = state_to_load['finalized_cases']
		self.total_cycle_time = state_to_load['total_cycle_time']
		self.current_state = state_to_load

	def run(self, running_time=RUNNING_TIME, auto=True, stop_at_plan_tasks=False):
    
		while self.now <= running_time:
			event = self.events.pop(0)
			self.now = event[0]
			event = event[1]
			if event.event_type == EventType.CASE_ARRIVAL:
				self.unassigned_tasks[event.task.id] = event.task
				self.case_start_times[event.task.case_id] = self.now
				self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.CASE_ARRIVAL))
				self.planner.report(Event(event.task.case_id, event.task, self.now, None, EventType.TASK_ACTIVATE))
				self.busy_cases[event.task.case_id] = [event.task.id]
				self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
				(t, task) = self.problem.next_case()
				self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))
				self.events.sort()
			elif event.event_type == EventType.START_TASK:
				self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.START_TASK))
				t = self.now + self.problem.processing_time_sample(event.resource, event.task)
				self.events.append((t, SimulationEvent(EventType.COMPLETE_TASK, t, event.task, event.resource)))
				self.events.sort()
				if not self.problem.is_event(event.task.task_type):
					del self.reserved_resources[event.resource]
					self.busy_resources[event.resource] = (event.task, self.now)
			elif event.event_type == EventType.COMPLETE_TASK:
				self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.COMPLETE_TASK))
				if not self.problem.is_event(event.task.task_type):  # for actual tasks (not events)
					del self.busy_resources[event.resource]
					if self.working_nr_resources() <= self.desired_nr_resources():
						self.available_resources.add(event.resource)
					else:
						self.away_resources.append(event.resource)
						self.away_resources_weights.append(
							self.problem.resource_weights[self.problem.resources.index(event.resource)])
				del self.assigned_tasks[event.task.id]
				self.busy_cases[event.task.case_id].remove(event.task.id)
				next_tasks = self.problem.complete_task(event.task)
				for next_task in next_tasks:
					self.unassigned_tasks[next_task.id] = next_task
					self.planner.report(Event(event.task.case_id, next_task, self.now, None, EventType.TASK_ACTIVATE))
					self.busy_cases[event.task.case_id].append(next_task.id)
				if len(self.busy_cases[event.task.case_id]) == 0:
					self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.COMPLETE_CASE))
					self.events.append((self.now, SimulationEvent(EventType.COMPLETE_CASE, self.now, event.task)))
				self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
				self.events.sort()
			elif event.event_type == EventType.SCHEDULE_RESOURCES:
				if len(self.away_resources) > 0:
					i = random.randrange(len(self.away_resources))
				required_resources = self.desired_nr_resources() - self.working_nr_resources()
				if required_resources > 0:
					for i in range(required_resources):
						random_resource = random.choices(self.away_resources, self.away_resources_weights)[0]
						away_resource_i = self.away_resources.index(random_resource)
						del self.away_resources[away_resource_i]
						del self.away_resources_weights[away_resource_i]
						self.available_resources.add(random_resource)
					self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
					self.events.sort()
				elif required_resources < 0:
					nr_resources_to_remove = min(len(self.available_resources), -required_resources)
					resources_to_remove = random.sample(self.available_resources, nr_resources_to_remove)
					for r in resources_to_remove:
						self.available_resources.remove(r)
						self.away_resources.append(r)
						self.away_resources_weights.append(
							self.problem.resource_weights[self.problem.resources.index(r)])
				self.events.append((self.now + 1, SimulationEvent(EventType.SCHEDULE_RESOURCES, self.now + 1, None)))
				self.events.sort()
			elif event.event_type == EventType.PLAN_TASKS:
				if len(self.unassigned_tasks) > 0 and len(self.available_resources) > 0:
					if stop_at_plan_tasks:
						self.events.insert(0, (self.now, event))
						break
					else:
					#assignments = self.planner.plan(self.scenario_tree, self.available_resources.copy(), list(self.unassigned_tasks.values()), self.problem_resource_pool)
						assignments = self.planner.plan_selected(self.available_resources.copy(), list(self.unassigned_tasks.values()), self.problem_resource_pool)
						moment = self.now
						print ('Assignments:', assignments)
						print('Unassigned tasks:', self.unassigned_tasks)
						for (task,resource) in assignments:
							if task not in self.unassigned_tasks.values():
								return None, "ERROR: trying to assign a task that is not in the unassigned_tasks."
							if resource not in self.available_resources:
								return None, "ERROR: trying to assign a resource that is not in available_resources."
							if resource not in self.problem_resource_pool[task.task_type]:
								return None, "ERROR: trying to assign a resource to a task that is not in its resource pool."
						if not auto:
							#print(assignments)
							if len(assignments) > 0:
								self.apply_user_selected_assignment(assignments, moment, event)
						else:
							self.handle_plan_tasks_event(event)
					
			elif event.event_type == EventType.COMPLETE_CASE:
				self.total_cycle_time += self.now - self.case_start_times[event.task.case_id]
				self.finalized_cases += 1
		unfinished_cases = 0
		for busy_tasks in self.busy_cases.values():
			if len(busy_tasks) > 0:
				if busy_tasks[0] in self.unassigned_tasks:
					busy_case_id = self.unassigned_tasks[busy_tasks[0]].case_id
				else:
					busy_case_id = self.assigned_tasks[busy_tasks[0]][0].case_id
				if busy_case_id in self.case_start_times:
					start_time = self.case_start_times[busy_case_id]
					if start_time <= running_time:
						self.total_cycle_time += running_time - start_time
						self.finalized_cases += 1
						unfinished_cases += 1
		return (self.total_cycle_time / self.finalized_cases, "COMPLETED: you completed " + str(running_time) + " hours of simulated customer cases. " + str(self.finalized_cases) + " cases started. " + str(self.finalized_cases - unfinished_cases) + " cases run to completion.", self.scenario_tree)

	def apply_user_selected_assignment(self, assignments, moment, event):
		# Displaying task-resource assignments for user selection
		print("Select which task-resource assignment to apply:")
		index = 0
		for (task,resource) in assignments:
			print(f"[{index}]: T{task.id}_{resource}")
			index += 1

		# User selection
		while True:
			user_input = input("Enter the index of the assignment to apply: ").strip()
			if user_input.isdigit() and 0 <= int(user_input) < len(assignments):
				selected_index = int(user_input)
				task,resource = assignments[selected_index]
				self.apply_assignment(task, resource, moment, event)
				break
			else:
				print("Invalid index, try again.")
    
	def load_parent_state(self, parent_state):
		self.now = parent_state['now']
		self.events = parent_state['events']
		self.unassigned_tasks = parent_state['unassigned_tasks']
		self.assigned_tasks = parent_state['assigned_tasks']
		self.available_resources = parent_state['available_resources']
		self.busy_resources = parent_state['busy_resources']
		self.reserved_resources = parent_state['reserved_resources']
		self.busy_cases = parent_state['busy_cases']
		self.away_resources = parent_state['away_resources']
		self.away_resources_weights = parent_state['away_resources_weights']
		self.finalized_cases = parent_state['finalized_cases']
		self.total_cycle_time = parent_state['total_cycle_time']
		self.current_state = parent_state
  
	def save_current_state(self):
		self.current_state = {
			'now': self.now,
			'events': self.events,
			'unassigned_tasks': self.unassigned_tasks,
			'assigned_tasks': self.assigned_tasks,
			'available_resources': self.available_resources,
			'busy_resources': self.busy_resources,
			'reserved_resources': self.reserved_resources,
			'busy_cases': self.busy_cases,
            'away_resources': self.away_resources,
            'away_resources_weights': self.away_resources_weights,
            'finalized_cases': self.finalized_cases,
            'total_cycle_time': self.total_cycle_time
		}
		return self.current_state

	def apply_assignment(self, task, resource, moment, event):
			if task.id in self.unassigned_tasks:
				del self.unassigned_tasks[task.id]
			else:
				print("ERROR: trying to assign a task that is not in the unassigned_tasks.")
			self.assigned_tasks[task.id] = (task, resource, moment)
			if not self.problem.is_event(task.task_type):
				self.available_resources.remove(resource)
				self.reserved_resources[resource] = (event.task, moment)
			assignment = (task, resource)
			child_node = self.scenario_tree.generate_child_node(self.scenario_tree.current_node, assignment, self.available_resources.copy(), self.now)
			self.scenario_tree.current_node = child_node
			self.current_node = child_node
			self.events.sort()
   
	def handle_plan_tasks_event(self, event):
		assignments = self.planner.plan_selected(self.available_resources, list(self.unassigned_tasks.values()), self.problem_resource_pool)

		# Save the current state and node for backtracking
		parent_state = self.save_current_state()
		parent_node = self.current_node

		for (task, resource) in assignments:
			# Apply the task-resource assignment, which also updates the current_node
			self.apply_assignment(task, resource, self.now, event)

			# Continue the simulation from this new state until the next PLAN_TASKS event
			self.run(auto=True, stop_at_plan_tasks=True)

			# Backtrack to the previous state and node to explore other branches
			self.load_parent_state(parent_state)
			self.current_node = parent_node
			self.scenario_tree.current_node = parent_node