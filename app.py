
import streamlit as st
import torch
import gymnasium as gym
import time
from PIL import Image

from a2c_model import Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="A2C CartPole Live Demo", layout="wide")
st.title("🎮 A2C CartPole – Démonstration Live")

AGENTS = {
    "Agent 0 – Baseline": "models/agent0.pt",
    "Agent 1 – Reward Masking": "models/agent1.pt",
    "Agent 2 – Parallel (K=6)": "models/agent2.pt",
    "Agent 3 – n-step": "models/agent3.pt",
    "Agent 4 – Batch (K=6,n=6)": "models/agent4.pt",
}

st.sidebar.header("⚙️ Paramètres")
agent_name = st.sidebar.selectbox("Choisir l'agent", list(AGENTS.keys()))
episodes = st.sidebar.slider("Nombre d'épisodes", 1, 5, 1)
fps = st.sidebar.slider("FPS", 5, 60, 30)
policy_mode = st.sidebar.radio("Politique", ["Greedy (argmax)", "Stochastique"])
start = st.sidebar.button("▶️ Lancer la démo")

@st.cache_resource
def load_agent(path):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    actor = Actor(state_dim, action_dim).to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE))
    actor.eval()
    return actor

if start:
    actor = load_agent(AGENTS[agent_name])
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    frame_box = st.empty()
    info_box = st.empty()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0
        step = 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                dist = actor(state_t)
                if policy_mode == "Greedy (argmax)":
                    action = torch.argmax(dist.probs).item()
                else:
                    action = dist.sample().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            step += 1

            frame = env.render()
            frame_box.image(Image.fromarray(frame), width=600)

            info_box.markdown(
                f"**Agent**: {agent_name}  \n"
                f"**Épisode**: {ep+1}/{episodes}  \n"
                f"**Step**: {step}  \n"
                f"**Return**: {ep_return}"
            )

            time.sleep(1 / fps)

        st.success(f"Épisode {ep+1} terminé – Return = {ep_return}")

    env.close()
