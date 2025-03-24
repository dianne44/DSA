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

num_cities = distances.shape[0]
num_neurons = num_cities * 2  # More neurons than cities for flexibility
iterations = 5000
learning_rate = 0.8
neighborhood_size = num_neurons // 2

# Initialize neurons randomly with city indices, ensuring valid moves
neurons = np.random.choice(num_cities, num_neurons, replace=True)

# Gaussian neighborhood function
def neighborhood(winner_idx, size):
    distances = np.abs(np.arange(num_neurons) - winner_idx)
    return np.exp(-(distances ** 2) / (2 * (size ** 2)))

# Training loop
for i in range(iterations):
    city_idx = np.random.randint(num_cities)  # Pick a random city
    neuron_distances = np.array([distances[city_idx, n] for n in neurons])

    # Ignore infinite distances (unreachable cities)
    valid_indices = np.where(neuron_distances != np.inf)[0]
    if len(valid_indices) == 0:
        continue  # Skip if no valid connections

    winner_idx = valid_indices[np.argmin(neuron_distances[valid_indices])]

    # Update neurons towards the selected city (but only if a valid connection exists)
    influence = neighborhood(winner_idx, neighborhood_size)
    for j in range(len(neurons)):
        if distances[neurons[j], city_idx] != np.inf:  # Ensure valid moves
            neurons[j] = city_idx if np.random.rand() < influence[j] else neurons[j]

    # Decay learning rate and neighborhood size
    learning_rate *= 0.9997
    neighborhood_size *= 0.9997
    neighborhood_size = max(neighborhood_size, 1)

# Extract final TSP route from trained neurons (convert to Python integers)
final_route = []
for neuron in neurons:
    if int(neuron) not in final_route:
        final_route.append(int(neuron))

# Ensure all cities are visited and form a valid cycle
missing_cities = set(range(num_cities)) - set(final_route)
for city in missing_cities:
    final_route.append(city)  # Add missing cities to ensure a valid cycle

final_route.append(final_route[0])  # Return to starting city

# Compute total route distance, ensuring no invalid (inf) connections
total_distance = 0
valid_route = True

for i in range(len(final_route) - 1):
    d = distances[final_route[i], final_route[i+1]]
    if d == np.inf:
        valid_route = False  # Mark the route as invalid if an inf distance exists
        break
    total_distance += d

# If route is invalid, regenerate a valid cycle using nearest-neighbor heuristic
if not valid_route:
    print("Invalid route detected! Generating a valid TSP solution using Nearest Neighbor...")

    # Nearest-Neighbor Fix: Start at city 0 and build a valid path
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

    valid_route.append(0)  # Return to start
    final_route = valid_route

    # Recalculate distance
    total_distance = sum(distances[final_route[i], final_route[i+1]] for i in range(len(final_route)-1))

# Print results
print("Final TSP Route:", final_route)
print("Total Distance:", total_distance)

# Plot the route
plt.figure(figsize=(6, 6))
plt.title("SOM TSP Route (Adjacency Matrix)")
for i in range(len(final_route) - 1):
    plt.plot([i, i+1], [final_route[i], final_route[i+1]], 'bo-')

plt.xticks(range(num_cities), labels=[f'City {i+1}' for i in range(num_cities)])
plt.yticks(range(num_cities), labels=[f'City {i+1}' for i in range(num_cities)])
plt.grid()
plt.show()
