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

    # Load graph
    with open("data/graph_900_1800_3600.pkl", "rb") as f:
        G = pickle.load(f)

    if args.outage:
        # Remove all the edges from 2 stations to simulate an outage
        utils.remove_all_trains(G, from_station="St-Maurice", to_station="Martigny")

    # Convert start and destination to location (lon, lat)
    start_loc = utils.get_location(G, args.start)
    end_loc = utils.get_location(G, args.end)

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
    mode_penalties = utils.get_penalties(args.sustainability)
    print(f"Mode penalties: {mode_penalties}")
    dists, edges_to = utils.dijkstra(
        G,
        "Start",
        "End",
        start_time=start_time,
        change_penalty=args.change_penalty,
        mode_penalties=mode_penalties,
    )
    print(dists)

    # Reconstruct path
    edges = utils.reconstruct_edges(edges_to, "Start", "End")
    for edge in edges[::-1][1:]:
        print(edge)

    # Postprocess path
    path = utils.postprocess_path(edges[:-1])

    # Print journey
    utils.pretty_print(path, args)


if __name__ == "__main__":
    main()
