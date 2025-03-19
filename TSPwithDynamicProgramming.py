def solve_tsp_dynamic(distances):
    n = len(distances)
    dp = {}

    # Initialize base case
    for i in range(1, n):
        if distances[0][i] != float('inf'):
            dp[(i, 1 << i | 1)] = distances[0][i]  # Include city 0 in mask

    # Solve for all subsets
    for mask in range(1, 1 << n):
        if (mask & 1) == 0:  # Must include city 0
            continue

        for end in range(n):
            if (mask & (1 << end)) == 0:
                continue

            prev_mask = mask & ~(1 << end)
            min_distance = float('inf')

            for prev in range(n):
                if (prev_mask & (1 << prev)) == 0 or prev == end:
                    continue
                if (prev, prev_mask) in dp and distances[prev][end] != float('inf'):
                    dist = dp[(prev, prev_mask)] + distances[prev][end]
                    min_distance = min(min_distance, dist)

            if min_distance != float('inf'):
                dp[(end, mask)] = min_distance

    # Find shortest tour
    min_total_distance = float('inf')
    all_cities_mask = (1 << n) - 1
    last_city = -1

    for i in range(1, n):
        if (i, all_cities_mask) in dp and distances[i][0] != float('inf'):
            total_distance = dp[(i, all_cities_mask)] + distances[i][0]
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                last_city = i  # Store last city

    if min_total_distance == float('inf'):
        return [], float('inf')  # No valid tour

    # ✅ FIX: Ensure last_city is valid before path reconstruction
    if last_city == -1:
        return [], float('inf')

    # Reconstruct path
    path = [0]
    mask = all_cities_mask
    current = last_city

    while current != 0:
        path.append(current)
        prev_city = -1

        for i in range(n):
            if (mask & (1 << i)) and i != current and (i, mask & ~(1 << current)) in dp:
                if dp[(i, mask & ~(1 << current))] + distances[i][current] == dp[(current, mask)]:
                    prev_city = i
                    break

        mask = mask & ~(1 << current)
        current = prev_city

        # ✅ FIX: If prev_city remains -1, break to avoid infinite loop
        if prev_city == -1:
            break

    path.append(0)
    path.reverse()
    return [p + 1 for p in path], min_total_distance

# Adjacency matrix representation
distances = [
    [0, 10, 12, float('inf'), float('inf'), float('inf'), 12],
    [10, 0, 8, 12, float('inf'), float('inf'), float('inf')],
    [12, 8, 0, 11, 3, float('inf'), 9],
    [float('inf'), 12, 11, 0, 11, 10, float('inf')],
    [float('inf'), float('inf'), 3, 11, 0, 6, 7],
    [float('inf'), float('inf'), float('inf'), 10, 6, 0, 9],
    [12, float('inf'), 9, float('inf'), 7, 9, 0]
]

path, min_distance = solve_tsp_dynamic(distances)
print("Optimal Path:", path)
print("Minimum Distance:", min_distance)
