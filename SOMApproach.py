import numpy as np
import matplotlib.pyplot as plt

# Define adjacency matrix (TSP distances between cities)
distances = np.array([
    [0, 10, 12, np.inf, np.inf, np.inf, 12],
    [10, 0, 8, 12, np.inf, np.inf, np.inf],
    [12, 8, 0, 11, 3, np.inf, 9],
    [np.inf, 12, 11, 0, 11, 10, np.inf],
    [np.inf, np.inf, 3, 11, 0, 6, 7],
    [np.inf, np.inf, np.inf, 10, 6, 0, 9],
    [12, np.inf, 9, np.inf, 7, 9, 0]
])

num_cities = distances.shape[0] #This determines the number of cities from the adjacency matrix distances.
num_neurons = num_cities * 2 #This sets the number of neurons in the Self-Organizing Map (SOM). 
iterations = 5000 #This sets the number of training iterations for the SOM algorithm.
learning_rate = 0.8
neighborhood_size = num_neurons // 2

#Assign neurons randomly to cities
neurons = np.random.choice(num_cities, num_neurons, replace=True)

# Gaussian neighborhood function
def neighborhood(winner_idx, size):
    distances = np.abs(np.arange(num_neurons) - winner_idx)
    return np.exp(-(distances ** 2) / (2 * (size ** 2)))

# Training loop
for i in range(iterations):
    city_idx = np.random.randint(num_cities)  
    neuron_distances = np.array([distances[city_idx, n] for n in neurons])

    # Ignore infinite distances
    valid_indices = np.where(neuron_distances != np.inf)[0]
    if len(valid_indices) == 0:
        continue  

    winner_idx = valid_indices[np.argmin(neuron_distances[valid_indices])]

    # Update neurons
    influence = neighborhood(winner_idx, neighborhood_size)
    for j in range(len(neurons)):
        if distances[neurons[j], city_idx] != np.inf:
            neurons[j] = city_idx if np.random.rand() < influence[j] else neurons[j]

    learning_rate *= 0.9997
    neighborhood_size *= 0.9997
    neighborhood_size = max(neighborhood_size, 1)

# Extract final route, ensuring unique cities
final_route = []
visited = set()
for city in neurons:
    if city not in visited:
        final_route.append(city)
        visited.add(city)

# Ensure all cities appear exactly once
missing_cities = set(range(num_cities)) - set(final_route)

# Insert missing cities at best positions
for city in missing_cities:
    best_position = None
    best_cost = float('inf')

    for i in range(len(final_route) - 1):
        city_a, city_b = final_route[i], final_route[i + 1]

        # Only insert if valid connections exist
        if distances[city_a, city] != np.inf and distances[city, city_b] != np.inf:
            insertion_cost = distances[city_a, city] + distances[city, city_b] - distances[city_a, city_b]
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_position = i + 1

    # Insert the missing city at the best position found
    if best_position is not None:
        final_route.insert(best_position, city)
    else:
        print(f" WARNING: City {city} could not be inserted due to missing connections!")

#  Ensure city 6 is in the route
if 6 not in final_route:
    print(" WARNING: City 6 is missing! Forcing it into the route.")
    final_route.append(6)

# Ensure the route forms a cycle
if final_route[0] != final_route[-1]:
    final_route.append(final_route[0])

#  Validate route connections
valid_route = True
for i in range(len(final_route) - 1):
    d = distances[final_route[i], final_route[i+1]]
    print(f"Distance from {final_route[i]} to {final_route[i+1]}: {d}")

    if d == np.inf:
        valid_route = False

#  If invalid, regenerate using Nearest-Neighbor heuristic
if not valid_route:
    print("Invalid route detected! Using Nearest Neighbor...")
    unvisited = set(range(1, num_cities))
    valid_route = [0]

    while unvisited:
        last = valid_route[-1]
        nearest_city = min(unvisited, key=lambda c: distances[last, c] if distances[last, c] != np.inf else np.inf)
        
        if distances[last, nearest_city] == np.inf:
            total_distance = np.inf
            print("No fully connected route found.")
            break

        valid_route.append(nearest_city)
        unvisited.remove(nearest_city)

    valid_route.append(0)  
    final_route = valid_route

# Convert to integers
final_route = [int(city) for city in final_route]

# Compute total distance
total_distance = sum(
    distances[final_route[i], final_route[i+1]] for i in range(len(final_route)-1)
)

# Print results
print("Final TSP Route:", final_route)
print("Total Distance:", total_distance)

# Plot the route
plt.figure(figsize=(6, 6))
plt.title("SOM TSP Route (Corrected)")

# Plot city connections
for i in range(len(final_route) - 1):
    plt.plot([i, i+1], [final_route[i], final_route[i+1]], 'bo-')
# Convert indexes to human-readable city numbers (1-based)
human_readable_route = [city + 1 for city in final_route]

print("Final TSP Route (City Numbers):", human_readable_route)
print("Total Distance:", total_distance)

plt.xticks(range(num_cities), labels=[f'City {i+1}' for i in range(num_cities)])
plt.yticks(range(num_cities), labels=[f'City {i+1}' for i in range(num_cities)])
plt.grid()
plt.show()
