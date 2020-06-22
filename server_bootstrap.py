"""Entrypoint that can be used to start many inference workers on a single node."""

import covid19sim.inference.server_bootstrap


def main(args=None):
    """
    [summary]

    Args:
        args ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    return covid19sim.inference.server_bootstrap.main(args)


if __name__ == "__main__":
    main()
