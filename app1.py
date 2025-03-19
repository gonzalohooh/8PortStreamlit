import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy
import os



def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        stored_username = os.getenv("STREAMLIT_USERNAME", "admin")
        stored_password = os.getenv("STREAMLIT_PASSWORD", "password")
        
        if st.button("Login"):
            if username == stored_username and password == stored_password:
                st.session_state.authenticated = True
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

authenticate()


def summarize_trial_pokes_weighted(trial_pokes, n_ports=8, first_poke_weight=2.0):
    if not trial_pokes:
        return 0  
    weighted_pokes = [trial_pokes[0]] * int(first_poke_weight) + list(trial_pokes)
    poke_counts = np.bincount(weighted_pokes, minlength=n_ports+1)[1:]
    poke_probs = poke_counts / np.sum(poke_counts)
    return entropy(poke_probs, base=2) if np.sum(poke_counts) > 0 else 0

def initialize_belief_model_1(kappa=0.5, w_today=0.4, w_yesterday=0.2, w_exploration=0.004):
    n_ports, n_trials = 8, 20
    thetas = np.linspace(0, 2*np.pi, n_ports+1)[:-1]
    f_r_today = np.exp(kappa * np.cos(thetas))
    f_r_today /= f_r_today.sum()
    f_r_yesterday = np.exp(kappa * np.cos(thetas))
    f_r_yesterday /= f_r_yesterday.sum()
    f_r_exploration = np.exp(0.015 * np.cos(thetas))
    f_r_exploration /= f_r_exploration.sum()
    f_r = w_today * f_r_today + w_yesterday * f_r_yesterday + w_exploration * f_r_exploration
    f_r /= f_r.sum()
    tau_true = np.random.randint(1, n_trials+1)
    return np.outer(f_r, np.ones(n_trials)), tau_true, f_r_today, f_r_yesterday, f_r_exploration

def plot_probability_distributions(w_today, w_yesterday, w_exploration, kappa):

    thetas = np.linspace(0, 2*np.pi, 8+1)[:-1]

    f_r_today = np.exp(kappa * np.cos(thetas - thetas[3]))
    f_r_today /= f_r_today.sum()
    
    f_r_yesterday = np.exp(kappa * np.cos(thetas - thetas[3]))
    f_r_yesterday /= f_r_yesterday.sum()
    
    f_r_exploration = np.exp(0.015 * np.cos(thetas - thetas[3]))
    f_r_exploration /= f_r_exploration.sum()

    plt.figure(figsize=(8, 4))
    plt.plot(thetas, w_today * f_r_today, '--', label="Today's Prior", alpha=0.6)
    plt.plot(thetas, w_yesterday *f_r_yesterday, '--', label="Yesterday's Prior", alpha=0.6)
    plt.plot(thetas, w_exploration * f_r_exploration, '--', label="Exploration Prior", alpha=0.6)
    plt.xlabel("Angle (radians)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Mixture of Three Von Mises Priors")
    st.pyplot(plt.gcf(), True)

def generate_pomdp_data(n_trials=20):
    belief, r_true, tau_true, last_visit, prev_r_true = initialize_belief()
    observations = []
    trial_markers = []
    belief_over_trials = []
    pokes_per_trial_dist = [1, 2, 3]  # Distribution of pokes per trial
    
    for t in range(n_trials):
        max_pokes = np.random.choice(pokes_per_trial_dist)
        for _ in range(max_pokes):
            action = np.argmax(belief.sum(axis=1))
            observations.append(action)
            trial_markers.append(t)
            belief_over_trials.append(belief.copy())
            w = int((action == r_true) and (t >= tau_true))
            belief, _, _ = update_belief(belief, w, action, t, last_visit)
            if w:
                return observations, trial_markers, belief_over_trials, r_true, prev_r_true
    
    return observations, trial_markers, belief_over_trials, r_true, prev_r_true



def initialize_belief(kappa=0.5, w_today=0.4, w_yesterday=0.2, w_exploration=0.004):
    n_ports = 8
    n_trials = 20
    thetas = np.linspace(0, 2*np.pi, n_ports+1)[:-1]
    
    r_true_today = np.random.randint(0, n_ports)
    r_true_yesterday = np.random.randint(0, n_ports)
    
    f_r_today = np.exp(kappa * np.cos(thetas - thetas[r_true_today]))
    f_r_today /= f_r_today.sum()
    
    f_r_yesterday = np.exp(kappa * np.cos(thetas - thetas[r_true_yesterday]))
    f_r_yesterday /= f_r_yesterday.sum()
    
    f_r_exploration = np.exp(0.015 * np.cos(thetas - thetas[r_true_today]))
    f_r_exploration /= f_r_exploration.sum()
    
    f_r = w_today * f_r_today + w_yesterday * f_r_yesterday + w_exploration * f_r_exploration
    f_r /= f_r.sum()
    
    tau_true = np.random.randint(1, n_trials+1)
    g_tau = np.ones(n_trials)
    
    b_prior = np.outer(f_r, g_tau)
    last_visit = np.zeros(n_ports, dtype=int)
    
    return b_prior, r_true_today, tau_true, last_visit, r_true_yesterday



def determine_state(belief, correct_port, yesterday_port):
    max_port = np.argmax(belief.sum(axis=1))
    if max_port == correct_port:
        return "Correct Port"
    elif max_port == yesterday_port:
        return "Yesterday Port"
    else:
        return "Exploring"

def get_start_prob(belief, correct_port, yesterday_port):
    total_belief = belief.sum(axis=1)
    start_probs = np.zeros(3)
    start_probs[0] = np.sum(total_belief) / len(total_belief)
    start_probs[1] = total_belief[yesterday_port]
    start_probs[2] = total_belief[correct_port]
    return start_probs / np.sum(start_probs)

def estimate_transition_prob(state_sequence):
    num_states = 3
    trans_counts = np.zeros((num_states, num_states))
    for t in range(len(state_sequence) - 1):
        from_state = states.index(state_sequence[t])
        to_state = states.index(state_sequence[t + 1])
        trans_counts[from_state, to_state] += 1
    return np.nan_to_num(trans_counts / trans_counts.sum(axis=1, keepdims=True))

def estimate_emission_prob(observations, state_sequence):
    num_states = 3
    num_ports = 8
    emit_counts = np.zeros((num_states, num_ports))
    for t, obs in enumerate(observations):
        state_idx = states.index(state_sequence[t])
        emit_counts[state_idx, obs] += 1
    return np.nan_to_num(emit_counts / emit_counts.sum(axis=1, keepdims=True))



def update_belief(b, w, p, t, last_visit):
    n_ports, n_trials = b.shape
    b_new = np.copy(b)
    reward_likelihood = (t >= np.arange(n_trials)) * (p == np.arange(n_ports)[:, None])
    if w == 1:
        b_new[:, :] = 0
        b_new[p, :] = reward_likelihood[p, :]
    else:
        b_new *= (1 - reward_likelihood)
    return b_new / np.sum(b_new) if np.sum(b_new) > 0 else b_new, None, last_visit


def viterbi_algorithm(observations, states, start_prob, trans_prob, emit_prob):
    num_states = len(states)
    num_obs = len(observations)
    
    V = np.zeros((num_states, num_obs))
    path = np.zeros((num_states, num_obs), dtype=int)
    
    for s in range(num_states):
        V[s, 0] = start_prob[s] * emit_prob[s, observations[0]]
    
    for t in range(1, num_obs):
        for s in range(num_states):
            prob_transitions = V[:, t - 1] * trans_prob[:, s]
            best_prev_state = np.argmax(prob_transitions)
            V[s, t] = prob_transitions[best_prev_state] * emit_prob[s, observations[t]]
            path[s, t] = best_prev_state
    
    best_final_state = np.argmax(V[:, -1])
    best_path = [best_final_state]
    
    for t in range(num_obs - 1, 0, -1):
        best_final_state = path[best_final_state, t]
        best_path.insert(0, best_final_state)
    
    return [states[i] for i in best_path]


def run_multiple_episodes(n_episodes, num_trials, w_today, w_yesterday, w_exploration, model):
    all_observations = []
    for _ in range(n_episodes):
        observations, _, _, _, _ = generate_pomdp_data(num_trials)
        all_observations.append(observations)
    return all_observations

def plot_experiment_summary(all_observations):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, obs in enumerate(all_observations):
        ax.plot(obs, alpha=0.6, linestyle='-', marker='o', label=f'Episode {i+1}')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Chosen Ports')
    ax.set_title('Complete Experiment Summary')
    ax.legend()
    st.pyplot(fig)


import torch
import torch.nn as nn
import random
from scipy.stats import entropy
from torchsummary import summary
import graphviz

class SimpleRLModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=8):
        super(SimpleRLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def display_nn_architecture():
    st.subheader("Neural Network Architecture")
    model = SimpleRLModel()
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')
    dot.node("Input Layer", shape='box', style='filled', fillcolor='lightblue')
    dot.node("Hidden Layer (16 Neurons)", shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.node("Output Layer", shape='box', style='filled', fillcolor='lightcoral')
    dot.edge("Input Layer", "Hidden Layer (16 Neurons)")
    dot.edge("Hidden Layer (16 Neurons)", "Output Layer")
    st.graphviz_chart(dot)



st.title('POMDP Memory Strategy Simulation')
model_choice = st.selectbox('Select POMDP Model', ['Model 1', 'RL Model'])

if model_choice == 'RL Model':
    display_nn_architecture()

num_trials = st.slider('Number of Trials', 10, 50, 20)
n_episodes = st.slider('Number of Episodes', 1, 10, 3)
w_today = st.slider('Weight for Today Memory', 0.0, 1.0, 0.4, 0.05)
w_yesterday = st.slider('Weight for Yesterday Memory', 0.0, 1.0, 0.2, 0.05)
w_exploration = st.slider('Weight for Exploration', 0.0, 0.05, 0.004, 0.001)
kappa = st.slider('Kappa Value', 0.0, 10.0, 0.5, 0.1)





plot_probability_distributions(w_today, w_yesterday, w_exploration, kappa)

if st.button('Run Simulation'):
    st.session_state.all_observations = run_multiple_episodes(n_episodes, num_trials, w_today, w_yesterday, w_exploration, model_choice)
    st.session_state.simulation_ran = True

if 'simulation_ran' in st.session_state and st.session_state.simulation_ran:
    st.subheader("Experiment Summary")
    plot_experiment_summary(st.session_state.all_observations)
    
    trial_view = st.selectbox("Select a trial to view details", range(1, n_episodes+1))
    selected_obs = st.session_state.all_observations[trial_view - 1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(selected_obs, marker='o', linestyle='-', label=f'Episode {trial_view}')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Chosen Ports')
    ax.set_title(f'Detailed View of Episode {trial_view}')
    ax.legend()
    st.pyplot(fig)
