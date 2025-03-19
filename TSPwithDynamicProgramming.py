def solve_tsp_dynamic(distances):
    n = len(distances)
    # Initialize the memoization table with infinity
    # dp[i][mask] represents the shortest path distance ending at city i and visiting cities in mask
    dp = {}
    
    # Set up the base case: distance from start (city 0) to each city
    for i in range(1, n):
        if distances[0][i] != float('inf'):
            dp[(i, 1 << i)] = distances[0][i]
    
    # Iterate through all subsets of cities
    for mask in range(1, 1 << n):
        # Skip if the subset doesn't include the starting city
        if mask & (1 << 0) == 0:
            continue
            
        for end in range(n):
            # Skip if the end city is not in the subset
            if (mask & (1 << end)) == 0:
                continue
                
            # Skip if we're only looking at a single city
            if mask == (1 << end):
                continue
                
            # Calculate the subset without the end city
            prev_mask = mask & ~(1 << end)
            
            # Find the best path to the end city
            min_distance = float('inf')
            for prev in range(n):
                if (prev_mask & (1 << prev)) == 0 or prev == end:
                    continue
                if (prev, prev_mask) in dp and distances[prev][end] != float('inf'):
                    dist = dp[(prev, prev_mask)] + distances[prev][end]
                    min_distance = min(min_distance, dist)
            
            if min_distance != float('inf'):
                dp[(end, mask)] = min_distance
    
    # Find the shortest path that visits all cities and returns to the start
    min_total_distance = float('inf')
    all_cities_mask = (1 << n) - 1
    
    for last_city in range(1, n):
        if (last_city, all_cities_mask) in dp and distances[last_city][0] != float('inf'):
            total_distance = dp[(last_city, all_cities_mask)] + distances[last_city][0]
            min_total_distance = min(min_total_distance, total_distance)
    
    # Reconstruct the path
    path = [0]  # Start with city 0
    mask = all_cities_mask
    current = None
    
    # Find the last city in the optimal path
    for i in range(1, n):
        if (i, mask) in dp and distances[i][0] != float('inf'):
            if dp[(i, mask)] + distances[i][0] == min_total_distance:
                current = i
                break
    
    # Reconstruct the rest of the path backwards
    path.append(current)
    mask = mask & ~(1 << current)
    
    while mask > 1:  # Stop when only the starting city is left
        for i in range(n):
            if (mask & (1 << i)) and (i, mask) in dp and distances[i][current] != float('inf'):
                if dp[(i, mask)] + distances[i][current] == dp[(current, mask | (1 << current))]:
                    path.append(i)
                    mask = mask & ~(1 << i)
                    current = i
                    break
    
    path.append(0)  # Return to the starting city
    path.reverse()  # Reverse to get the correct order
    
    # Convert to 1-indexed for output
    path_1_indexed = [p + 1 for p in path]
    
    return path_1_indexed, min_total_distance
