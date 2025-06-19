import matplotlib

matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import math
from flask import Flask, render_template, request, Request
import io
import base64


app = Flask(__name__)


# Graph class with A* implementation
class Graph:
    def __init__(self, fuel_capacity, fuel_consumption_per_km):
        self.graph = nx.Graph()
        self.fuel_capacity = fuel_capacity
        self.fuel_consumption_per_km = fuel_consumption_per_km

    def add_node(self, node, pos):
        """Add a node with a position for visualization"""
        self.graph.add_node(node, pos=pos)

    def add_edge(self, u, v, weight=1):
        """Add an edge from vertex u to v with the given weight"""
        self.graph.add_edge(u, v, weight=weight)

    def visualize(self, path=None):
        """Visualize the graph and the path"""
        pos = nx.get_node_attributes(self.graph, "pos")
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=700,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
        )

        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        if path:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=path_edges, edge_color="red", width=2
            )

        plt.title("City Graph with Shortest Path", fontsize=14)
        plt.axis("off")

        # Save the plot to a BytesIO object and convert to base64
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode("utf8")

    def heuristic(self, node, end):
        """Heuristic function for A* (Euclidean distance)"""
        pos = nx.get_node_attributes(self.graph, "pos")
        x1, y1 = pos[node]
        x2, y2 = pos[end]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def a_star(self, start, end):
        if start == end:
            return [start], 0, 0  # No distance or fuel consumed

        priority_queue = [
            (0, start, self.fuel_capacity)
        ]  # (total_cost, current_node, remaining_fuel)
        visited = set()
        cost_so_far = {start: 0}
        came_from = {start: None}

        while priority_queue:
            total_cost, current_node, remaining_fuel = heapq.heappop(priority_queue)

            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.reverse()
                fuel_consumed = self.fuel_capacity - remaining_fuel
                return path, cost_so_far[end], fuel_consumed

            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor in self.graph.neighbors(current_node):
                edge_weight = self.graph[current_node][neighbor]["weight"]
                fuel_needed = edge_weight * self.fuel_consumption_per_km

                # Refuel if current node is a petrol bunk
                if current_node in ["G_Bunk", "H_Bunk"]:
                    remaining_fuel = self.fuel_capacity

                # Check if there's enough fuel to move to the neighbor
                if remaining_fuel >= fuel_needed:
                    new_cost = cost_so_far[current_node] + edge_weight
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        came_from[neighbor] = current_node
                        heapq.heappush(
                            priority_queue,
                            (
                                new_cost + self.heuristic(neighbor, end),
                                neighbor,
                                remaining_fuel - fuel_needed,
                            ),
                        )

        return None, float("inf"), 0


# Initialize the graph with nodes and edges
g = Graph(fuel_capacity=15, fuel_consumption_per_km=0.25)

g.add_node("A", pos=(0, 0))  # Starting point
g.add_node("B", pos=(2, 2))  # Intermediate stop
g.add_node("C", pos=(5, 1))  # Intermediate stop
g.add_node("D", pos=(7, 4))  # Intermediate stop
g.add_node("E", pos=(10, 2))  # Destination
g.add_node("F", pos=(4, 5))  # Intermediate stop
g.add_node("G_Bunk", pos=(6, 6))  # Petrol bunk
g.add_node("H_Bunk", pos=(9, 5))  # Petrol bunk
g.add_node("I", pos=(3, -3))  # Intermediate stop
g.add_node("J", pos=(8, -2))  # Intermediate stop
g.add_node("K", pos=(12, 4))  # Final destination

# Adding edges with distances (in kilometers)
edges = [
    ("A", "B", 30),
    ("A", "I", 56),
    ("B", "C", 2),
    ("B", "F", 14),
    ("C", "D", 30),
    ("C", "I", 4),
    ("D", "E", 4),
    ("D", "F", 5),
    ("D", "G_Bunk", 20),
    ("E", "H_Bunk", 22),
    ("E", "K", 3),
    ("F", "G_Bunk", 33),
    ("G_Bunk", "H_Bunk", 42),
    ("H_Bunk", "K", 5),
    ("I", "J", 6),
    ("J", "E", 3),
    ("J", "K", 4),
]

for u, v, weight in edges:
    g.add_edge(u, v, weight=weight)


@app.route("/", methods=["GET", "POST"])
def index():
    img_data = None
    path = None
    distance = None
    fuel_consumed = None

    if request.method == "POST":
        start = request.form.get("start")
        end = request.form.get("end")
        if start in g.graph.nodes and end in g.graph.nodes:
            path, distance, fuel_consumed = g.a_star(start, end)
            if path:
                img_data = g.visualize(path=path)
            else:
                path = "No valid path found due to insufficient fuel or no available route."
        else:
            path = "Invalid start or end node."

    return render_template(
        "index.html",
        nodes=g.graph.nodes,
        img_data=img_data,
        path=path,
        distance=distance,
        fuel_consumed=fuel_consumed,
    )


# For Vercel Python Serverless Function
def handler(request: Request):
    return app(request.environ, start_response=lambda *args: None)
