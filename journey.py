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

    print(
        utils.get_best_path(
            start=args.start,
            end=args.end,
            date=args.date,
            time=args.time,
            limit=args.limit,
            sustainability=args.sustainability,
            outage=args.outage,
        )
    )


if __name__ == "__main__":
    main()
