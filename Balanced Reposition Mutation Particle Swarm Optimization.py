import numpy as np

# Define the Rastrigin function
def rastrigin(x):
    n = len(x)
    return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# Define the Sphere function
def sphere(x):
    return sum(xi**2 for xi in x)


# Define the PSO algorithm
def pso(cost_func, dim=2, num_particles=30, max_iter=500, w=0.5, c1=1, c2=2, mutation_rate = 1):
    
    iterator = 0 #An iterator variable to keep track of the number of iterations for PSO

    # Initialize particles and velocities
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitnesses = []
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Initialize the accumulating lists of centers of masses and their fitnesses
    centers = []
    com_fitnesses = []

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):

        # Calculate the center of mass
        xcom = (sum(particles[x][0] for x in range(num_particles)))/num_particles
        ycom = (sum(particles[x][1] for x in range(num_particles)))/num_particles
        com = [xcom, ycom]
        centers.append(com)
        com_fitness = cost_func(com)
        com_fitnesses.append(com_fitness)
        
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (com - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)
        best_fitnesses.append(np.min(fitness_values))

        # Perform a check to see if there is an improvement in particle fitnesses and update inertia weight
        if iterator > 1:
            if best_fitnesses[iterator] < best_fitnesses[iterator - 1]:
                w *= 1.1
            else:
                w *= 0.9
        
        # Apply a reposition mutation to a random dimension of a particle
        for particle in range(num_particles):
            if np.random.rand() < mutation_rate:
                dimension = np.random.choice(dim)
                particles[particle, dimension] = 0

        iterator += 1    
    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness
    

# Define the dimensions of the problem
dim = 2

# Run the PSO algorithm on the test functions
solution_rastrigin, fitness_rastrigin = pso(rastrigin, dim=dim)
solution_sphere, fitness_sphere = pso(sphere, dim=dim)

# Print the solution and fitness values for test functions
print('Solution rastrigin:', str(solution_rastrigin), "\nFitness rastrigin:", str(fitness_rastrigin))
print('\n\nSolution sphere:', str(solution_sphere), "\nFitness sphere:", str(fitness_sphere))