# Understanding Generative Flow Networks (GFlowNets) – A User Perspective

This repository provides an interactive demonstration of Generative Flow Networks (GFlowNets) with:

- A **static web interface** (`index.html`) showcasing:
  - An introduction to GFlowNets
  - An interactive **Tetris** demo guided by a pretrained GFlowNet policy
  - Visualizations of core concepts and flow conservation in DAGs (e.g., molecule generation)
  - Sections on advanced mathematics and further reading
- **Pretraining scripts** (`tetris_agent.py`) to train a Tetris GFlowNet using a neural network policy
- **Static assets**:
  - **`static/`**: JavaScript files powering comparison charts, flow conservation demos, and molecule DAG rendering
  - **`molecules/`**: Images used in the molecule flow visualization

---

## Features

- **GFlowNet Overview**: Clear, interactive explanation of GFlowNets, comparison to traditional RL, and theoretical underpinnings
- **Tetris Demo**: Real-time Tetris game where a GFlowNet samples and balances multiple candidate moves
- **Flow Visualizations**: Animated DAGs illustrating flow conservation and distribution across solutions
- **Advanced Math**: Expandable sections with formal definitions (Flow Matching, Trajectory Balance, Detailed Balance)
- **Pretraining Script**: Python code to train your own GFlowNet on Tetris using partial rewards and penalties

## Prerequisites

- **Web Browser** (Chrome, Firefox, Edge)
- **Python 3.8+**

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/gflownet-demo.git
   cd gflownet-demo
