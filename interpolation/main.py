import argparse
import logging
import interpolation.suite as suite
import interpolation.interface as interface


def parse_inputs():
    parser = argparse.ArgumentParser(description=
                                     """Interpolation and optimisation of SHARPy reduced order models
                                     """)

    parser.add_argument('input_file', help='input file in YAML format')
    parser.add_argument("-r", "--refresh", action="store_true",
                        help="Refresh inputs (remove training_data.txt and testing_data.txt)")
    parser.add_argument('-o', '--optimize', action='store_true',
                        help='Run optimiser on interpolation')

    parser = parser.parse_args()

    return parser


def main():
    parser = parse_inputs()

    settings = interface.load_yaml(parser.input_file)

    if parser.optimize:
        opti = OptimisedInterpolation(parser.input_file)
        opti.optimise()
    else:
        interpolation_suite = suite.Suite(parser.input_file, refresh=parser.refresh)

        # interpolation_suite.run_training()
        # interpolation_suite.run_testing()

        interpolation_suite.interpolate()
        # interpolation_suite.evaluate()
        interpolation_suite.teardown()

    return 0
