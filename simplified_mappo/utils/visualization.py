import matplotlib.pyplot as plt
import numpy as np

def render_env(env, ax=None, agent_trails=None):
    """
    Render the environment state using matplotlib
    """
    if ax is None:
        _, ax = plt.subplots()
    
    # Access world through env.world
    world = env.world
    
    # Initialize agent_trails if not provided
    if agent_trails is None:
        agent_trails = {f'agent_{i}': [] for i in range(len(world.agents))}
    
    # Plot landmarks
    landmark_pos = np.array([l.state.p_pos for l in world.landmarks])
    ax.scatter(landmark_pos[:, 0], landmark_pos[:, 1], c='gray', s=100, label='Landmarks')
    
    # Plot agents and update trails
    agent_pos = np.array([agent.state.p_pos for agent in world.agents])
    agent_vel = np.array([agent.state.p_vel for agent in world.agents])
    
    # Calculate distances between agents and landmarks
    distances = []
    for agent_idx, agent_pos_single in enumerate(agent_pos):
        # Calculate distances to all landmarks for this agent
        agent_distances = np.linalg.norm(landmark_pos - agent_pos_single, axis=1)
        distances.append(agent_distances)
        
        # Update trail
        agent_trails[f'agent_{agent_idx}'].append(agent_pos_single)
        
        # Plot trail
        trail = np.array(agent_trails[f'agent_{agent_idx}'])
        ax.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.5)
        
        # Draw line to closest landmark
        closest_landmark_idx = np.argmin(agent_distances)
        closest_landmark_pos = landmark_pos[closest_landmark_idx]
        ax.plot([agent_pos_single[0], closest_landmark_pos[0]], 
                [agent_pos_single[1], closest_landmark_pos[1]], 
                'k--', alpha=0.3)
    
    # Plot agent positions
    ax.scatter(agent_pos[:, 0], agent_pos[:, 1], c='blue', s=100, label='Agents')
    
    # Plot velocity vectors
    ax.quiver(agent_pos[:, 0], agent_pos[:, 1], 
             agent_vel[:, 0], agent_vel[:, 1], 
             color='red', scale=20, width=0.005)
    
    # Set plot limits and labels
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend()
    
    return distances, agent_trails
    