#  Pathfinding with Fuel Constraints – AI Project

This is a simple AI-powered pathfinding simulation that introduces **fuel constraints** into the routing logic. The project demonstrates how a pathfinding algorithm (e.g., A*, Dijkstra) adapts when an agent is limited by fuel and must plan accordingly.

##  Project Structure

| File/Folder      | Description |
|------------------|-------------|
| `app.py`         | Main Flask application that handles route logic and serves the frontend. |
| `templates/`     | Contains HTML templates (e.g., `index.html`) for rendering the UI.        |
| `index.html`     | The main web interface for interacting with the pathfinding system.       |
| `README.md`      | Project documentation (you're reading it).                               |

## Features

- **AI Pathfinding:** Uses intelligent algorithms to compute the shortest or optimal path.
- **Fuel Constraint:** Incorporates fuel consumption logic to affect decisions.
- **Web Interface:** Clean and minimal frontend for user interaction.
- **Fast Response:** Designed to provide real-time or near real-time feedback.
- **Extendable:** Can be integrated with maps, obstacles, refueling stations, etc.

## 🖥️ How to Run

### 🔧 Prerequisites

- Python 3.x
- Flask

### Installation

```bash
git clone https://github.com/yourusername/pathfinding-with-fuel.git
cd pathfinding-with-fuel
pip install flask
```
### TO RUN THE FILE
```bash
python app.py
```
-- Now open the browser on: 
*http://127.0.0.1:5000*
