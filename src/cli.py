import argparse


def manage_args():
    parser = argparse.ArgumentParser(description="""CM Learning""")

    parser.add_argument(
        "-a",
        "--algorithm-name",
        help="Learning Algorithm name",
        default="xgboost",
        dest="alg",
        type=str,
    )

    args = parser.parse_args()
    return args


def main():
    args = manage_args()


if __name__ == "__main__":
    main()
