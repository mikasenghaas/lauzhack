"""
Main entry point for computing journey.
"""
import numpy as np
import utils as utils
import pandas as pd
import pickle


def main():
    # Parse arguments
    args = utils.get_args()
    print(args)

    # Load graph
    with open("data/graph_900_900_900.pkl", "rb") as f:
        G = pickle.load(f)

    print(len(G.nodes), len(G.edges))

    if not args.intermodal:
        raise ValueError("Only intermodal journeys are supported")

    # Convert start and destination to location (lon, lat)
    start_loc = utils.get_location(args.start)
    end_loc = utils.get_location(args.end)

    # Add both locations to graph
    G.add_node("Start", pos=start_loc)
    G.add_node("End", pos=end_loc)

    # Find k-closest stations from start and end
    dists_from_start = []
    dists_to_end = []
    for station, attr in G.nodes(data=True):
        try:
            station_pos = attr["pos"]
            dists_from_start.append(
                (station, utils.get_distance(start_loc, station_pos))
            )
            dists_to_end.append((station, utils.get_distance(end_loc, station_pos)))
        except Exception as e:
            continue

    # Sort the distances in place
    start_k_closest = sorted(dists_from_start, key=lambda x: x[1])[: args.limit]
    end_k_closest = sorted(dists_to_end, key=lambda x: x[1])[: args.limit]

    # Compute travel time from start to k closest stations
    for mode in ["foot", "bike", "car"]:
        for station, dist in start_k_closest:
            if args.exact_travel_time:
                travel_time = utils.get_exact_travel_time(
                    start_loc, station, method=mode
                )
                G.add_edge("Start", station, duration=travel_time, type=mode)
            else:
                travel_time = utils.get_approx_travel_time(dist, method=mode)
                G.add_edge("Start", station, duration=travel_time, type=mode)

        for station, dist in end_k_closest:
            if args.exact_travel_time:
                travel_time = utils.get_exact_travel_time(dist, method=mode)
                G.add_edge(station, "End", duration=travel_time, type=mode)
            else:
                travel_time = utils.get_approx_travel_time(dist, method=mode)
                G.add_edge(station, "End", duration=travel_time, type=mode)

    # Run Dijkstra on graph
    start_time = pd.to_datetime(f"{args.date} {args.time}")
    print(start_time)
    dists, edges_to = utils.dijkstra(G, "Start", "End", start_time=start_time)

    # Reconstruct path
    path = utils.reconstruct_path(edges_to, "Start", "End")

    # Postprocess path
    # processed_path = utils.postprocess_path(path)

    print(path)


if __name__ == "__main__":
    main()
