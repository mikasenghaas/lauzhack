import numpy as np
import argparse
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
import datetime

import utils as utils


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-foot-travel", default=900, type=int)
    parser.add_argument("--max-bike-travel", default=900, type=int)
    parser.add_argument("--max-car-travel", default=900, type=int)

    args = parser.parse_args()

    return args


def read_timetable():
    # Load the data
    df = pd.read_csv("data/ist-daten-sbb.csv", sep=";")

    # Select and rename columns
    cols = {
        "Journey identifier": "journey_id",
        "Arrival time": "arrival",
        "Departure time": "departure",
        "Stop name": "station",
        "OPUIC": "opuic",
        "Geopos": "pos",
    }

    # Select and rename columns
    df = df[cols.keys()].rename(columns=cols)

    # Convert time columns to datetime
    df["arrival"] = pd.to_datetime(df["arrival"])
    df["departure"] = pd.to_datetime(df["departure"])

    return df


def read_pr_stations():
    # Read in city and regional stations
    df = pd.read_csv("data/pr_stations.csv")

    # Rename columns
    df = df.rename(columns={"station": "name", "station_abbr": "abbr"})

    return df


def build_edge_list(df):
    # Build edge list (with edge attributes) for train journeys
    edges = []
    for journey_id in tqdm(df.journey_id.unique()):
        trip = df[df.journey_id == journey_id].sort_values("departure", inplace=False)
        trip_name = f"{trip.iloc[0].station} -> {trip.iloc[-1].station}"

        for i in range(len(trip) - 1):
            edges.append(
                (
                    trip.iloc[i].station,
                    trip.iloc[i + 1].station,
                    {
                        "departure": trip.iloc[i].departure,
                        "arrival": trip.iloc[i + 1].arrival,
                        "duration": trip.iloc[i + 1].arrival - trip.iloc[i].departure,
                        "journey_id": journey_id,
                        "trip_name": trip_name,
                        "type": "train",
                    },
                )
            )

    return edges


def main():
    # Args
    args = get_args()

    # Read timetable and P+R station data
    print("Reading timetable and P+R station data...")
    timetable = read_timetable()
    pr = read_pr_stations()

    print("Building edge list")
    edges = build_edge_list(timetable)

    # Create graph
    G = nx.MultiDiGraph(edges)

    # Add node attributes (id, pos, has_pr)
    print("Adding node attributes...")
    unique_stations = timetable.drop_duplicates(subset=["station"])
    node_attrs = {
        row.station: {
            "opuic": row.opuic,
            "pos": row.pos,
            "has_pr": row.station in pr.name.values,
        }
        for _, row in unique_stations.iterrows()
    }

    nx.set_node_attributes(G, node_attrs)

    limits = {
        "foot": args.max_foot_travel,  # s (30min)
        "bike": args.max_bike_travel,  # s (1h)
        "car": args.max_car_travel,  # s (1h)
    }

    # Iterate over all pairs of nodes and mode types (V*V*num_modes)
    added_edges = {
        "foot": 0,
        "bike": 0,
        "car": 0,
    }
    print("Growing graph for foot, bike and car modalities...")
    for u, attr_u in tqdm(G.nodes(data=True)):
        for v, attr_v in G.nodes(data=True):
            # Discard if same node
            if u == v:
                continue

            # Get longitude and latitude for start and end station
            u_pos = attr_u["pos"]
            v_pos = attr_v["pos"]

            # Discard if no position available
            if u_pos is np.nan or v_pos is np.nan:
                continue

            for mode in ["foot", "bike", "car"]:
                # Discard if air distance is above thresholds (not reachable with mode)
                dist = utils.get_distance(u_pos, v_pos)  # dist in m
                if dist > limits[mode] * utils.AVG_SPEED[mode]:
                    continue

                # Otherwise compute travel time for mode and add edge if below threshold
                time = utils.get_approx_travel_time(dist, mode)
                if time < limits[mode]:
                    # print(f"Adding from {u} to {v} via {mode} in {(time/60):.2f}min (less than {limits[mode]/60:.2f}min)")
                    G.add_edge(
                        u, v, mode=mode, duration=datetime.timedelta(seconds=time)
                    )
                    added_edges[mode] += 1

    print(f"Added {added_edges['foot']} foot edges.")
    print(f"Added {added_edges['bike']} bike edges.")
    print(f"Added {added_edges['car']} car edges.")

    # Save graph
    out_path = f"data/graph_{args.max_foot_travel}_{args.max_bike_travel}_{args.max_car_travel}.pkl"
    print(f"Saving graph to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(G, f)


if __name__ == "__main__":
    main()
