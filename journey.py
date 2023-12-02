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

        # 1. Find k-closest stations with PR
        # 2.1 Compute distance from start to each station
        # 2.2 Compute journey (top-k) from each possible start to end
        # 3. Combine journeys
        # 4. Rank journey and return top-k
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
