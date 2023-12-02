"""
Main entry point for computing journey.
"""
import utils as utils
import pandas as pd
import pySBB as sbb


def main():
    # Parse arguments
    args = utils.get_args()
    print(args)

    # Load PR stations
    pr_stations = pd.read_csv("data/pr_stations.csv")

    if args.intermodal:
        # Convert address to location
        start_loc = utils.get_location(args.start)
        print(f"Start location: {start_loc}")

        # 1. Find k-closest stations with PR
        distances = [
            (end[0], utils.get_distance(start_loc, f"{end[2]}, {end[3]}"))
            for _, end in pr_stations.iterrows()
        ]
        # for i in distances:
        #    print(i)

        k = args.limit
        distances.sort(key=lambda t: t[1])
        k_closest_stations = distances[:k]

        # 2.1 Compute distance from start to each station in seconds
        car_distances = [
            (
                station[0],
                utils.getRouteTime(
                    start_loc, utils.get_location(station[0]), method="foot"
                ),
            )
            for station in k_closest_stations
        ]

        # 2.2 Compute journey (top-k) from each possible start to end
        travels = []
        for station, dist in car_distances:
            print(f"Computing journey from {station} to {args.end}")
            connections = sbb.get_connections(
                station,
                args.end,
                via=args.via,
                date=args.date,
                time=args.time,
                transportations=args.transportations,
                limit=1,
            )
            for connection in connections:
                # print(connection)
                travels.append(
                    {
                        "start": args.start,
                        "station": station,
                        "seconds_to_station": dist,
                        "end": args.end,
                        "seconds_from_station_to_end": connection.duration.seconds,
                        "connection": connection,
                    }
                )

        # 3. Combine journeys
        # 4. Rank journey and return top-k
        travels.sort(
            key=lambda t: t["seconds_to_station"] + t["seconds_from_station_to_end"]
        )
        best_travel = travels[0]

        print(
            f"The fastes journey is:\n"
            f"1) Drive from {best_travel['start']} to {best_travel['station']} in {best_travel['seconds_to_station']} s\n"
            f"2) Take the train from {best_travel['station']} to {best_travel['end']} in {best_travel['seconds_from_station_to_end']} s\n"
            f"\t{best_travel['connection']}\n"
            f"Total time: {best_travel['seconds_to_station'] + best_travel['seconds_from_station_to_end']} s"
        )

        pass

    else:
        # Compute journey from start to end
        connections = sbb.get_connections(
            args.start,
            args.end,
            via=args.via,
            date=args.date,
            time=args.time,
            transportations=args.transportations,
            limit=args.limit,
        )

        for connection in connections:
            print(connection)


if __name__ == "__main__":
    main()
