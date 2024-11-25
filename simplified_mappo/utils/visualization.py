import matplotlib.pyplot as plt
import numpy as np

def render_env(env, ax=None):
    """
    Render the environment state using matplotlib
    """
    if ax is None:
        _, ax = plt.subplots()
    
    # Access world through env.world
    world = env.world
    
    # Plot landmarks
    landmark_pos = np.array([l.state.p_pos for l in world.landmarks])
    ax.scatter(landmark_pos[:, 0], landmark_pos[:, 1], c='gray', s=100, label='Landmarks')
    
    # Plot agents
    agent_pos = np.array([agent.state.p_pos for agent in world.agents])
    agent_vel = np.array([agent.state.p_vel for agent in world.agents])
    
    # Plot agent positions
    ax.scatter(agent_pos[:, 0], agent_pos[:, 1], c='blue', s=100, label='Agents')
    
    # Plot velocity vectors
    ax.quiver(agent_pos[:, 0], agent_pos[:, 1], 
             agent_vel[:, 0], agent_vel[:, 1], 
             color='red', scale=20, width=0.005)
    
    # Draw lines between agents and closest landmarks
    for agent_idx, agent_pos in enumerate(agent_pos):
        # Find closest landmark
        distances = np.linalg.norm(landmark_pos - agent_pos, axis=1)
        closest_landmark_idx = np.argmin(distances)
        closest_landmark_pos = landmark_pos[closest_landmark_idx]
        
        # Draw line
        ax.plot([agent_pos[0], closest_landmark_pos[0]], 
                [agent_pos[1], closest_landmark_pos[1]], 
                'k--', alpha=0.3)
    
    # Set plot limits and labels
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend()
    
    return ax
    