import matplotlib.pyplot as plt
import numpy as np

def render_env(world, ax=None):
    """
    Render the environment using matplotlib
    
    Args:
        world: The MPE world object
        ax: Optional matplotlib axis to render on. If None, uses current axis
    """
    if ax is None:
        plt.clf()
        ax = plt.gca()
    
    # Plot landmarks
    landmark_pos = np.array([l.state.p_pos for l in world.landmarks])
    ax.scatter(landmark_pos[:, 0], landmark_pos[:, 1], c='gray', s=100, label='Landmarks')
    
    # Plot agents
    agent_pos = np.array([a.state.p_pos for a in world.agents])
    ax.scatter(agent_pos[:, 0], agent_pos[:, 1], c='blue', s=200, label='Agents')
    
    # Add velocity arrows
    for agent in world.agents:
        ax.arrow(agent.state.p_pos[0], agent.state.p_pos[1],
                agent.state.p_vel[0]*0.1, agent.state.p_vel[1]*0.1,
                head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.grid(True)
    plt.pause(0.01) 