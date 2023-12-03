"""
Some utility functions.
"""

import re
import csv
import argparse
import requests
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.distance import distance
import networkx as nx

import heapq

from datetime import date, datetime

PR_STATIONS = "./data/pr_stations"

# Average speed in m/s
AVG_SPEED = {
    "foot": 1.4,
    "bike": 4.17,
    "car": 13.89,
}

# Max travel time in seconds
MAX_TRAVEL_TIME = {
    "foot": 60 * 60 // 4,  # 15min
    "bike": 60 * 60 // 2,  # 30min
    "car": 60 * 60,  # 1h
}

PENALITIES = {
    "foot": 1,
    "bike": 1,
    "car": 10,
    "train": 1,
}


def get_args():
    parser = argparse.ArgumentParser()

    # Default date and time args
    # current_date = date.today().strftime("%Y-%m-%d")
    # current_time = datetime.now().strftime("%H:%M")
    current_date = "2023-12-01"
    current_time = "12:00"

    # Help messages
    loc_types = ["address", "station name", "station abbreviation", "coordinate"]
    transportation_types = ["train", "tram", "ship", "bus", "cableway"]
    start_help = f"Start location (Specify either {', '.join(loc_types)})"
    via_help = f"Locations to pass through (Specify either {', '.join(loc_types)})"
    stop_help = f"Stop location (Specify either {', '.join(loc_types)})"
    date_help = "Date of departure (Format: YYYY-MM-DD). Default: Today"
    time_help = "Time of departure (Format: YYYY-MM-DD). Default: Now"
    transportation_help = f"Modes of transportation (Specify from {', '.join(transportation_types)}). Default: All"

    # Specify line arguments
    parser.add_argument("--start", type=str, required=True, help=start_help)
    parser.add_argument("--via", type=list[str], help=via_help)
    parser.add_argument("--end", type=str, required=True, help=stop_help)
    parser.add_argument("--date", type=str, default=current_date, help=date_help)
    parser.add_argument("--time", type=str, default=current_time, help=time_help)
    parser.add_argument(
        "--limit", type=int, default=3, help="Number of journeys to return"
    )
    parser.add_argument(
        "--transportations",
        type=list[str],
        choices=transportation_types,
        default=["train"],
        help=transportation_help,
    )
    parser.add_argument("--exact-travel-time", action="store_true", help=time_help)
    parser.add_argument(
        "--intermodal", action="store_true", help="Only show intermodal journeys"
    )

    return parser.parse_args()


def get_location(address: str) -> str:
    """
    Converts an address to coordinates (latitude, longitude)

    Address can be:
        - Full name
        - Abbreviation (maximal 4 characters and uppercase)
        - Coordinates  (Format: "latitude, longitude")

    Args:
        address (str): Address to convert to coordinates.

    Returns:
        str: Coordinates of the address (Format: "latitude, longitude").
    """
    pattern = re.compile(r"^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$")

    # Check if it is already a coordinate
    if bool(pattern.match(address)):
        return address

    # Check if it is an abbreviation
    if len(address) <= 4 and address.isupper():
        with open(PR_STATIONS, mode="r") as file:
            reader = csv.reader(file)
            _ = next(reader)

            for row in reader:
                if row[1] == address:
                    latitude = row[3]
                    longitude = row[4]
                    return "{}, {}".format(latitude, longitude)

    # Use geopy to convert coordinates to address
    geolocator = Nominatim(user_agent="sbb_project")
    location = geolocator.geocode(
        address, country_codes="ch", language="de", exactly_one=True
    )

    return "{}, {}".format(location.latitude, location.longitude)


def get_distance(start, end):
    """
    Calculate distance between two coordinates in meters.
    Coordinates must be in the format "latitude, longitude".

    Args:
        start (str): Start coordinate.
        end (str): End coordinate.

    Returns:
        int: Distance between the two coordinates in meters.
    """
    start_converted = tuple(map(float, start.split(", ")))
    end_converted = tuple(map(float, end.split(", ")))
    return distance(start_converted, end_converted).meters


def get_exact_travel_time(start, end, method="car"):
    """
    Calculate travel time between two coordinates in seconds.
    """
    endpoint = "http://www.mapquestapi.com/directions/v2/route"
    if method not in ["foot", "bike", "car"]:
        raise ValueError("Method must be either foot, bike or car")

    names = {
        "foot": "pedestrian",
        "bike": "bicycle",
        "car": "fastest",
    }
    # Search params
    params = {
        "key": "HnDX3JAuALRTge28jbZVWO1L538fJbZE",
        "from": start,
        "to": end,
        "unit": "k",  # Use km instead of miles
        "narrativeType": "none",  # Just some other parameters to omit information we don't care about
        "sideOfStreetDisplay": False,
        "routeType": names[method],
    }

    # Do GET request and read JSON
    response = requests.get(endpoint, params=params)
    data = response.json()
    seconds = data["route"]["time"]

    return seconds


def get_approx_travel_time(dist, method="car"):
    """
    Calculate the approximate travel time between two coordinates in seconds.

    Args:
        dist (int): Distance between the two coordinates in meters.
        method (str, optional): Method of transportation. Defaults to "car".
    """
    return dist / AVG_SPEED[method]


def dijkstra(G, start, end, start_time, change_penalty=300):
    # Initialize distances and time dictionary with all distances set to infinity
    assert start in G.nodes(), f"Start node {start} not in graph"
    assert end in G.nodes(), f"End node {end} not in graph"

    distances = {
        node: {
            "distance": float("infinity"),
            "time": start_time,
        }
        for node in list(G.nodes())
    }

    # Initialise dictionary to keep track of the edges that lead to the node with the smallest distance
    edges_to = {node: None for node in list(G.nodes())}

    # Set the distance from the start node to itself to 0
    distances[start]["distance"] = 0
    edges_to[start] = ("", "Start", {"type": "foot", "journey_id": None, "duration": 0})

    # Priority queue to keep track of nodes with their current distances
    priority_queue = [(0, start)]

    while priority_queue:
        # Get the node with the smallest distance from the priority queue
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == end:
            return distances[end], edges_to

        # Check if the current distance is smaller than the stored distance
        if current_distance > distances[current_node]["distance"]:
            continue

        for _, neighbor, attributes in G.out_edges(current_node, data=True):
            no_duration_avail = not (type(attributes["duration"]) in [int, float])
            if no_duration_avail:
                continue

            # Initialise distance to neighbor
            distance = attributes["duration"]

            # Add penalty
            distance *= PENALITIES[attributes["type"]]

            # Compute wait time for train (if next edge is a train)
            wait = 0  # number of seconds you have to wait for the train
            next_is_train = attributes["type"] == "train"
            prev_is_train = edges_to[current_node][-1]["type"] == "train"
            changed_trip = False
            if next_is_train:
                wait = (
                    attributes["departure"] - distances[current_node]["time"]
                ).total_seconds()
                train_departed = (
                    distances[current_node]["time"] >= attributes["departure"]
                )

                if train_departed:
                    continue

                if prev_is_train:
                    changed_trip = (
                        edges_to[current_node][-1]["journey_id"]
                        != attributes["journey_id"]
                    )

            # Penalise changing and waiting times
            changed_mode = edges_to[current_node][-1]["type"] != attributes["type"]

            # Add max of waiting time or changing penalty to current dist
            if changed_mode or changed_trip:
                distance += max(wait, change_penalty)

            # Overall dist (prev dist + dist to neighbor)
            distance = current_distance + distance

            # If the new distance is smaller, update the distance and add to the priority queue
            if distance < distances[neighbor]["distance"]:
                # Update distance and time of arrival for neighbor
                time_of_arrival = distances[current_node]["time"] + pd.Timedelta(
                    seconds=distance
                )
                distances[neighbor]["distance"] = distance
                distances[neighbor]["time"] = time_of_arrival

                # Add edge that leads to neighbor
                edges_to[neighbor] = (current_node, neighbor, attributes)

                # Update priority queue
                heapq.heappush(priority_queue, (distance, neighbor))

    print(f"Probably there is no path between {start} and {end}")

    return distances, edges_to

def reconstruct_path(edges_to: list[tuple[str, str, dict]], start: str, end: str):
    path = [end]
    while path[-1] != start:
        path.append(edges_to[path[-1]][0])
        print(edges_to[path[-1]])
    return path[::-1]

def merge_edges(edges: list[tuple]) -> list[tuple]:
    """Merge two edges if they have same transport type.
    
        Edge Structure: (
           start, 
           end,
           {
               type,
               duration,
               departure  (only for train),
               arrival    (only for train),
               journey_id (only for train),
               trip_name  (only for train),
        )

    Args:
        edges (list[tuple]): list of travel edges.

    Returns:
        list[tuple]: post-processed list of travel edges.
    """
    traversed = []
    prev = None
    
    for edge in edges:
        
        if prev is None:
            prev = edge
            traversed.append(edge)
            continue
        
        # Need to check the transport type and, if it is train,
        # the journey id
        if edge[2]["type"] == prev[2]["type"] \
            and (edge[2]["type"] != "train" 
                 or edge[2]["journey_id"] == prev[2]["journey_id"]):
                
            prev = traversed.pop()
            
            # Merge the two edges
            new_edge = (
                prev[0],
                edge[1],
                {
                    "type": prev[2]["type"],
                    "duration": prev[2]["duration"] + edge[2]["duration"],
                },
            )
            
            if edge[2]["type"] == "train":
                new_edge[2]["departure"] = prev[2]["departure"]
                new_edge[2]["arrival"] = edge[2]["arrival"]
                new_edge[2]["journey_id"] = edge[2]["journey_id"]
                new_edge[2]["trip_name"] = edge[2]["trip_name"]
        else:
            new_edge = edge

            
        prev = new_edge
        traversed.append(new_edge)
            
    return traversed
