"""
Some utility functions.
"""

import re
import csv
import argparse
import requests
from pydoc import describe
import requests
import json
from geopy.geocoders import Nominatim
from geopy.distance import distance
import networkx as nx

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


def get_args():
    parser = argparse.ArgumentParser()

    # Default date and time args
    # current_date = date.today().strftime("%Y-%m-%d")
    # current_time = datetime.now().strftime("%H:%M")
    current_date = "2023-12-5"
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


def add_node(graph, node, k=5):
    """
    Adds a node to the graph if it doesn't exist yet.

    Args:
        graph (dict): Graph to add the node to.
        node (???): Node to add to the graph.
    """

    # Now we calculate the distance between the new node and all other nodes given the 3 different methods
    # We than take k-nearest neighbors and add them to the graph

    for method in ["foot", "bike", "car"]:
        distances = []
        for other_node in graph:
            if other_node == node:
                continue
            distance = getRouteTime(node, other_node, method)
            distances.append((distance, other_node, method))

        distances.sort(key=lambda x: x[0])
        graph[node].update(distance[:k])


def add_node(G: nx.Graph, start_loc: str, k: int) -> nx.Graph:
    """
    Adds a new node to the graph

    Args:
        G (nx.Graph): The graph
        start_loc (str): The starting location
        k (int): The number of closest stations to add

    Returns:
        nx.Graph: The graph with the new node
    """
