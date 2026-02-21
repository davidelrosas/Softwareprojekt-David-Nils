#What exactly is the hidden state, is it just another name for our hidden layer?

import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from jax import random

def main() -> int:
    key= random.key(int(time.time()))
    k1,k2 = random.split(key, 2)
    
    #generating synthetic data
    t0, t1 = 0.0, 10.0
    data_points = 100
    t, y_data = gen_synthetic_data(t0, t1, data_points, k1)
    
    #Initialize Neural network
    input_dim = 2 #a vector [y, t]
    hidden_dim = 16 #one neuron
    output_dim = 1 #dy/dt scalar
    params = init_params(input_dim, hidden_dim, output_dim, k2)
    
    #Training
    y0 = y_data[0]
    learning_rate = 0.01
    epochs = 200
    params = training_loop(params, y0, t, y_data, learning_rate, epochs)
    
    #prediction
    y_pred = euler_integration(params, y0, t)
    
    #Visualization
    plot_prediction(t, y_data, y_pred)
    return 0

def gen_synthetic_data(t0, t1, points_amt, random_key):
    """Returns points and corresponding y_data according to sin(t) + noise"""
    t = jnp.linspace(t0, t1, points_amt)
    noise_factor = 0.0
    k1, k2 = random.split(random_key, 2)
    #t = random.uniform(k1, shape = (points_amt,), minval = t0, maxval = t1)
    #jnp.sort(t)
    noise = random.normal(k2, shape = t.shape) * noise_factor
    
    y_data = jnp.sin(t) + noise
    return t, y_data


def init_params(input_dim, hidden_dim, output_dim, random_key):
    """Returns dictionary of weights and biases"""
    k1, k2 = jax.random.split(random_key, 2)
    params = {
        #W1: (in_dim, hidden_dim) so X@W1 -> (batch,hidden_dim)
        "W1": random.normal(k1, (input_dim, hidden_dim)),      
        "b1": jnp.zeros((hidden_dim,)),
        "W2": random.normal(k2, (hidden_dim, output_dim)),
        "b2": jnp.zeros((output_dim,))
    }
    
    return params

def forward(params, y, t):
    """Returns predicted dy/dt for given y and t"""
    X = jnp.stack([y, t])[None,:] #THE ISSUE WAS HERE
    
    #Hidden layer: activation via 𝜽2tanh((𝜽1^y)y + (𝜽1^t)t
    h = jnp.tanh(jnp.dot(X, params["W1"]) + params["b1"])   # (batch, hidden_dim)
    #why exactly are we usinh tanh?
    
    #output layer: linear activation for dy/dt 
    dy_dt = jnp.dot(h, params["W2"]) + params["b2"]# (batch, 1)
    #we try to approximate dy/dt = f𝜽 with the neural network! 
    # (The neural network is the function f𝜽, because when we call forward(params, y, t) we are computing f𝜽(t, y))
    return dy_dt.squeeze(-1) #we remove the last dimension and get shape (batch,), we output the prediction of the derivate f𝜽(t, y)

# 𝜽2tanh((𝜽1^y)y + (𝜽1^t)t means:
#   take y and t and apply a linear combination (𝜽1^y,𝜽1^t) then apply tanh()
#   then multiply by other set of weights 𝜽2, to get the scalar output

#y0 = y(t0)
#y1 = y0 + h𝜽2tanh((𝜽1^y)y0 + (𝜽1^t)t0)
#y2 = y1 + h𝜽2tanh((𝜽1^y)y1 + (𝜽1^t)t1)
def euler_integration(params, y0, t):   #here t is Δt for y(t+Δt)
    """Integrates dy/dt using neural network to get predicted y at all time points."""
    dt = t[1] - t[0]
    
    def step(y, t_curr):
        """we compute yn+1 using Euler's method"""
        dy_dt = forward(params, y, t_curr)
        y_next = y + dy_dt * dt 
        return jnp.squeeze(y_next), jnp.squeeze(y_next)

    _, y_pred = jax.lax.scan(step, y0, t[:-1])
    y_pred = jnp.concatenate([jnp.array([y0]), y_pred])
    return y_pred

def loss_fn(params, y0, t, y_data):
    """Returns MSE loss between predicted and true y"""
    y_pred = euler_integration(params, y0, t)   #predict y using Neural network + euler integration with initial value y0
    return jnp.mean((y_pred - y_data)**2)

def train_step(params, y0, t, y_data, lr):
    """Perform one gradient descent update on params"""
    grads = jax.grad(loss_fn)(params, y0, t, y_data) #computes ∂Loss/∂params for all parameters
    #update parameters using gradient descent
    new_params = {k: params[k] - lr * grads[k] for k in params}
    return new_params
    

def training_loop(params, y0, t, y_data, lr = 0.01, epochs = 200):
    """Iterate train_step for epochs iterations, returns trained params"""
    for epoch in range(epochs):
        params = train_step(params,y0, t, y_data, lr)
        
        if epoch % 10 == 0:
            current_loss = loss_fn(params, y0, t, y_data)
            print(f"Epoch {epoch:3d}: Loss = {current_loss:.4f}")
    print(f"Loss for trained NN: {loss_fn(params, y0, t, y_data)}")
    return params

def plot_prediction(t, y_data, y_pred):
    """Plots true vs predicted curve"""
    
    plt.figure(figsize=(8,5))
    plt.plot(t, y_data, label ="True curve", marker ='o')
    plt.plot(t, y_pred, label="predicted curve", marker='x')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("NODE: True vs Predicted Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())