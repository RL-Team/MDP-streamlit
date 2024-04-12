import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


# Streamlit UI
st.title("Markov Decision Process Example")

st.sidebar.title("MDP Parameters")
gamma = st.sidebar.slider("Discount Factor (γ)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

num_states = st.sidebar.slider("Number of States", min_value=2, max_value=6, value=2)
num_actions = st.sidebar.slider("Number of Actions", min_value=1, max_value=4, value=1)

# Initialize transition probabilities and rewards
transition_probs = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions, num_states))

st.sidebar.title("Define Transition Probabilities and Rewards")

# Define transition probabilities and rewards
for s in range(num_states):
    for a in range(num_actions):
        st.sidebar.subheader(f"State {s}, Action {a}")
        prob_sum = 0
        for s_prime in range(num_states):
            min_value=0.0
            max_value=1.0-prob_sum

            if (min_value == max_value) or (s_prime == num_states - 1):
                prob = max_value
                st.sidebar.write(f"T({s},{a},{s_prime}) = {prob:.2f}")
                st.sidebar.info(f"Transition probability is fixed at {prob:.2f}")
            else:
                prob = st.sidebar.slider(f"T({s},{a},{s_prime})", min_value=0.0, max_value=1.0-prob_sum, value=(1.0-prob_sum), step=0.01)
            transition_probs[s, a, s_prime] = prob
            prob_sum += prob

            rewards[s, a, s_prime] = st.sidebar.number_input(f"R({s},{a},{s_prime})", value=100.0)

# Calculate value iteration
values = np.zeros(num_states)
iterations = 0
while True:
    iterations += 1
    new_values = np.zeros(num_states)
    for s in range(num_states):
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for s_prime in range(num_states):
                action_values[a] += transition_probs[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * values[s_prime])
        new_values[s] = np.max(action_values)
    if (iterations > 100) or np.allclose(new_values, values):
        break
    values = new_values

# Display results
st.subheader("Optimal Values")
for s in range(num_states):
    st.write(f"V({s}) = {values[s]:.3f}")

st.subheader("Policy")
policy = np.zeros(num_states, dtype=int)
for s in range(num_states):
    action_values = np.zeros(num_actions)
    for a in range(num_actions):
        for s_prime in range(num_states):
            action_values[a] += transition_probs[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * values[s_prime])
    policy[s] = np.argmax(action_values)
    st.write(f"State {s}: Take Action {policy[s]}")