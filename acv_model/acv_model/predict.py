"""
Optional CLI wrapper, in case your organizers' predict.py / --input /
--output convention (mentioned in the ACV Info Kit but not the main
Problem Statement doc) turns out to be required. Confirm with organizers
whether this is actually needed.

Usage:
    python predict.py --input acv_test_case.xlsx --output acv_predictions.csv
"""
import argparse
from acv_model import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .xlsx file")
    parser.add_argument("--output", required=True, help="Path to write predictions.csv")
    args = parser.parse_args()

    filename = args.input.split("/")[-1].split("\\")[-1]
    result_df = predict(args.input, file_id=filename, output_format="dataframe")
    result_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
