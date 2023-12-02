"""
Some utility functions.
"""

import argparse
from pydoc import describe

from datetime import date, datetime


def get_args():
    parser = argparse.ArgumentParser()

    # Default date and time args
    current_date = date.today().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")

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
    parser.add_argument(
        "--intermodal", action="store_true", help="Only show intermodal journeys"
    )

    return parser.parse_args()
