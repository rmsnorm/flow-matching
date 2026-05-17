import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import flax.linen as nn
import optax
import logging
import matplotlib.animation as animation

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Chessboard distribution that we want to learn

#Sample data points from chessboard distribution
X_MIN = -4.0
X_MAX = 4.0
Y_MIN = -4.0
Y_MAX = 4.0
NUM_SQUARES_PER_AXIS = 4

def sample_chessboard_dist(n_samples=1000):
    total_squares = NUM_SQUARES_PER_AXIS ** 2
    # first select index of a black square
    index = np.random.choice(range(0, total_squares, 2), size=n_samples)
    index_y = index // NUM_SQUARES_PER_AXIS
    even_row = index_y % 2
    index_x = index % NUM_SQUARES_PER_AXIS + even_row

    # then sample x and y coordinate within the
    # black square.
    square_width = (X_MAX - X_MIN)/(NUM_SQUARES_PER_AXIS)
    x = np.random.uniform(0, square_width, size=n_samples)
    y = np.random.uniform(0, square_width, size=n_samples)

    final_coords_x = index_x * square_width + x + X_MIN
    final_coords_y = index_y * square_width + y + Y_MIN
    final_coords = np.hstack([final_coords_x[:, None], final_coords_y[:, None]])
    return final_coords
    

samples = sample_chessboard_dist(10000)

plt.scatter(samples[:, 0], samples[:, 1])

## Neural network that defines the vector field u_t(x)
class NeuralNet(nn.Module):
    d_f: int
    num_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray):
        # x is of shape [b, 2]
        z = jnp.hstack([x, t])
        for i in range(self.num_layers):
            z = nn.Dense(self.d_f)(z)
            z = nn.gelu(z)
        z = nn.Dense(2)(z)
        return z


# Define Train Step for Flow matching
def compute_loss(params, batch, vector_field, random_key):
    # sample a random time in [0, 1]
    key, subkey = jax.random.split(random_key)
    batch_size = batch.shape[0]
    t = jax.random.uniform(key, (batch_size, 1))

    # sample noise from normal distribution
    noise = jax.random.normal(subkey, shape=(batch_size, 2))

    z = batch
    x = t * z + (1.0 - t) * noise

    # compute loss
    # || vector_field.apply(params, x, t) - (z - noise) ||^2
    pred = vector_field.apply(params, x, t)
    loss = jnp.mean(jnp.sum((pred - (z - noise)) ** 2, axis=-1))
    return loss

def make_train_step(vector_field, optimizer):
    """Helper to build a JIT-compiled train_step closure."""
    @jax.jit
    def train_step(params, opt_state, batch, random_key):
        loss, grads = jax.value_and_grad(
            lambda p: compute_loss(p, batch, vector_field, random_key)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return train_step

# Define Euler's method for following the trajectory according to the vector field.

def euler_step(vector_field, x, t, h):
    t_input = jnp.full((x.shape[0], 1), t)
    return x + vector_field(x, t_input) * h

## Plot the trajectory of a randomly sampled point from p_init

def get_trajectory(vector_field, x, h, num_steps):
    trajectory = [x]
    for i in range(num_steps):
        t = i * h
        x = euler_step(vector_field, x, t, h)
        trajectory.append(x)
    return jnp.array(trajectory)

def plot_trajectory(trajectory):
    plt.plot(trajectory[..., 0], trajectory[..., 1])
    plt.show()

def plot_flow_trajectories(trajectories, num_plot=150):
    """Plots the actual trajectories of a subset of points flowing through time."""
    plt.figure(figsize=(6, 6))
    steps, samples, dims = trajectories.shape
    num_plot = min(num_plot, samples)
    
    # Plot starting points (noise at t=0) in blue
    plt.scatter(trajectories[0, :num_plot, 0], trajectories[0, :num_plot, 1], c='blue', alpha=0.5, s=10, label="Noise (t=0)")
    
    # Plot ending points (chessboard samples at t=1) in purple
    plt.scatter(trajectories[-1, :num_plot, 0], trajectories[-1, :num_plot, 1], c='purple', alpha=0.8, s=15, label="Chessboard (t=1)")
    
    # Plot paths connecting start and end
    for j in range(num_plot):
        plt.plot(trajectories[:, j, 0], trajectories[:, j, 1], color='gray', alpha=0.3, linewidth=1)
        
    plt.title("Trained Vector Field Flow Trajectories")
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_flow_trajectories(trajectories, num_plot=1000, filename="flow_animation.gif"):
    """Creates and saves an animation of points flowing through the vector field."""
    logging.info(f"Generating flow animation with {num_plot} particles...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    steps, samples, dims = trajectories.shape
    num_plot = min(num_plot, samples)
    
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_title("Flow Matching: Noise to Chessboard")
    ax.grid(True)
    
    # Initialize scatter plot
    scatter = ax.scatter(trajectories[0, :num_plot, 0], trajectories[0, :num_plot, 1], alpha=0.5, s=5, c='purple')
    
    def update(frame):
        scatter.set_offsets(trajectories[frame, :num_plot])
        ax.set_title(f"Flow Matching: Time Step {frame}/{steps-1}")
        return scatter,
        
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    ani.save(filename, writer='pillow', fps=20)
    logging.info(f"Animation saved to {filename}")
    plt.close(fig)

# Modular Helper Functions for Training and Sampling Flow Models

def initialize_model(vector_field, key):
    """Initializes model parameters for NeuralNet with correct dummy input shapes."""
    init_key, sample_key = jax.random.split(key)
    params = vector_field.init(init_key, jnp.zeros((1, 2)), jnp.zeros((1, 1)))
    return params, sample_key

def train_flow_matching(train_step, params, opt_state, key, num_iters=20000, batch_size=256):
    """Runs JIT-compiled optimization loop over chessboard distribution data."""
    logging.info("Starting training...")
    for i in range(num_iters + 1):
        batch = sample_chessboard_dist(batch_size)
        batch = jnp.array(batch)
        
        key, step_key = jax.random.split(key)
        params, opt_state, loss_val = train_step(params, opt_state, batch, step_key)
        
        if i % 500 == 0:
            logging.info(f"Iteration {i}: Loss = {loss_val:.4f}")
            
    logging.info("Training completed!")
    return params, opt_state

def sample_and_plot(vector_field, params, key, num_samples=10000, h=0.01, num_steps=100, generate_animation=False):
    """Generates novel points from the trained flow model and plots them."""
    vector_field_bound = vector_field.bind(params)
    
    # Generate new samples using the trained flow model!
    x_init = jax.random.normal(key, shape=(num_samples, 2))
    trajectories = get_trajectory(vector_field_bound, x_init, h=h, num_steps=num_steps)
    final_samples = trajectories[-1]
    
    # 1. Plot the actual flow trajectories
    logging.info("Plotting flow trajectories...")
    plot_flow_trajectories(trajectories, num_plot=150)
    
    # 2. Optionally create an animation
    if generate_animation:
        animate_flow_trajectories(trajectories, num_plot=2000, filename="flow_animation.gif")
        
    # 3. Plot the generated chessboard samples!
    logging.info("Plotting final generated chessboard samples...")
    plt.figure(figsize=(6, 6))
    plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, s=2, c='purple')
    plt.title("Generated Samples from Flow Matching")
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.grid(True)
    plt.show()
    return final_samples

if __name__ == "__main__":
    # 1. Initialize PRNGKey
    seed = 1
    key = jax.random.PRNGKey(seed)
    
    # 2. Instantiate Model and Optimizer
    num_iters = 20000
    # Swap to the more advanced FlowMatchingNet (you can easily revert to NeuralNet here)
    vector_field = FlowMatchingNet(64, 3)
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=num_iters)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-3)
    
    # 3. Create JIT-compiled training function
    train_step = make_train_step(vector_field, optimizer)
    
    # 4. Initialize Model Parameters & Optimizer State
    params, key = initialize_model(vector_field, key)
    opt_state = optimizer.init(params)
    
    # 5. Dry Run / Plot Initial Unoptimized Trajectory
    vector_field_bound = vector_field.bind(params)
    x = jax.random.normal(key, shape=(1, 2))
    trajectory = get_trajectory(vector_field_bound, x, h=0.05, num_steps=20)
    logging.info(f"Initial sample trajectory shape: {trajectory.shape}")
    sample_and_plot(vector_field, params, key, num_samples=10000, h=0.05, num_steps=20)
    
    # 6. Train the Flow Matching Model
    train_key, sample_key = jax.random.split(key)
    params, opt_state = train_flow_matching(train_step, params, opt_state, train_key, num_iters=num_iters, batch_size=1024)
    
    # 7. Sample and Plot Trained Results (smoother integration for animation)
    sample_and_plot(vector_field, params, sample_key, num_samples=10000, h=0.01, num_steps=100, generate_animation=True)


