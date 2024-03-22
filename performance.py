'''
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
    
'''   
   
   
       # Flag to control the monitoring thread
    #monitoring = True

    # Start the monitoring thread
    #monitor_thread = threading.Thread(target=monitor_resources, args=(0.01,))
    #monitor_thread.start()


    # Stop the monitoring thread
    #monitoring = False
    #monitor_thread.join()

    # Debugging prints
    #print(f"Data points collected: {len(memory_usage)}")
    #print(f"Memory usage data: {memory_usage}")
    #print(f"CPU usage data: {cpu_usage}")
   
   
   
   
   
'''print("Mean Timestamp: ", sum(timestamps)/len(timestamps))
    print("Max Timestamp: ", max(timestamps))
    print("Mean CPU usage: ", sum(cpu_usage)/len(cpu_usage))
    print("Max CPU usage: ", max(cpu_usage)) 
    print("Mean memory usage: ", sum(memory_usage)/len(memory_usage))
    print("Max memory usage: ", max(memory_usage))
     

    # Plot the resource usage
    #plot_resources()'''