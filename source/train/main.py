"""DeePMD-Kit entry point module."""

import argparse
from pathlib import Path

from deepmd.loggers import set_log_handles

from .config import config
from .doc import doc_train_input
from .freeze import freeze
from .test import test
from .train import train
from .transform import transform
from .compress import compress
from .doc import doc_train_input


def main():
    """DeePMD-Kit entry point.

    Raises
    ------
    RuntimeError
        if no command was input
    """
    parser = argparse.ArgumentParser(
        description="DeePMD-kit: A deep learning package for many-body potential energy"
        " representation and molecular dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # * logging options parser *********************************************************
    # with use of the parent argument this options will be added to every parser
    parser_log = argparse.ArgumentParser(add_help=False)
    parser_log.add_argument(
        "-v",
        "--verbose",
        default=2,
        action="count",
        dest="log_level",
        help="set verbosity level 0 - 3, 0=ERROR, 1(-v)=WARNING, 2(-vv)=INFO "
        "and 3(-vvv)=DEBUG",
    )
    parser_log.add_argument(
        "-l",
        "--log-path",
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
        "only be output to console",
    )

    # * config script ******************************************************************
    parser_cfig = subparsers.add_parser(
        "config",
        parents=[parser_log],
        help="fast configuration of parameter file for smooth model",
    )
    parser_cfig.add_argument(
        "-o", "--output", type=str, default="input.json", help="the output json file"
    )

    # * transform script ***************************************************************
    parser_transform = subparsers.add_parser(
        "transform", parents=[parser_log], help="pass parameters to another model"
    )
    parser_transform.add_argument(
        "-r",
        "--raw-model",
        default="raw_frozen_model.pb",
        type=str,
        help="the model receiving parameters",
    )
    parser_transform.add_argument(
        "-O",
        "--old-model",
        default="old_frozen_model.pb",
        type=str,
        help="the model providing parameters",
    )
    parser_transform.add_argument(
        "-o",
        "--output",
        default="frozen_model.pb",
        type=str,
        help="the model after passing parameters",
    )

    # * config parser ******************************************************************
    parser_train = subparsers.add_parser(
        "train", parents=[parser_log], help="train a model"
    )
    parser_train.add_argument(
        "INPUT", help="the input parameter file in json or yaml format"
    )
    parser_train.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=False,
        help="Initialize the model by the provided checkpoint.",
    )
    parser_train.add_argument(
        "-r",
        "--restart",
        type=str,
        default=False,
        help="Restart the training from the provided checkpoint.",
    )
    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.json",
        help="The output file of the parameters used in training.",
    )
    parser_train.add_argument(
        "-m",
        "--mpi-log",
        type=str,
        default="master",
        choices=("master", "collect", "workers"),
        help="Set the manner of logging when running with MPI. 'master' logs only on "
        "main process, 'collect' broadcasts logs from workers to master and 'workers' "
        "means each process will output its own log",
    )

    # * freeze script ******************************************************************
    parser_frz = subparsers.add_parser(
        "freeze", parents=[parser_log], help="freeze the model"
    )
    parser_frz.add_argument(
        "-c",
        "--checkpoint-folder",
        type=str,
        default=".",
        help="path to checkpoint folder",
    )
    parser_frz.add_argument(
        "-o",
        "--output",
        type=str,
        default="frozen_model.pb",
        help="name of graph, will output to the checkpoint folder",
    )
    parser_frz.add_argument(
        "-n",
        "--nodes",
        type=str,
        help="the frozen nodes, if not set, determined from the model type",
    )

    # * test script ********************************************************************
    parser_tst = subparsers.add_parser(
        "test", parents=[parser_log], help="test the model"
    )
    parser_tst.add_argument(
        "-m",
        "--model",
        default="frozen_model.pb",
        type=str,
        help="Frozen model file to import",
    )
    parser_tst.add_argument(
        "-s",
        "--system",
        default=".",
        type=str,
        help="The system dir. Recursively detect systems in this directory",
    )
    parser_tst.add_argument(
        "-S", "--set-prefix", default="set", type=str, help="The set prefix"
    )
    parser_tst.add_argument(
        "-n", "--numb-test", default=100, type=int, help="The number of data for test"
    )
    parser_tst.add_argument("-r", "--rand-seed", type=int, help="The random seed")
    parser_tst.add_argument(
        "--shuffle-test", action="store_true", help="Shuffle test data"
    )
    parser_tst.add_argument(
        "-d",
        "--detail-file",
        type=str,
        help="The file containing details of energy force and virial accuracy",
    )
    parser_tst.add_argument(
        "-a",
        "--atomic-energy",
        action="store_true",
        help="Test the accuracy of atomic energy",
    )

    # * compress model *****************************************************************
    # Compress a model, which including tabulating the embedding-net.
    # The table is composed of fifth-order polynomial coefficients and is assembled
    # from two sub-tables. The first table takes the stride(parameter) as it's uniform
    # stride, while the second table takes 10 * stride as it\s uniform stride
    #  The range of the first table is automatically detected by deepmd-kit, while the
    # second table ranges from the first table's upper boundary(upper) to the
    # extrapolate(parameter) * upper.
    parser_compress = subparsers.add_parser("compress", help="compress a model")
    parser_compress.add_argument(
        "INPUT",
        help="The input parameter file in json or yaml format, which should be "
        "consistent with the original model parameter file",
    )
    parser_compress.add_argument(
        "-i",
        "--input",
        default="frozen_model.pb",
        type=str,
        help="The original frozen model, which will be compressed by the deepmd-kit",
    )
    parser_compress.add_argument(
        "-o",
        "--output",
        default="frozen_model_compress.pb",
        type=str,
        help="The compressed model",
    )
    parser_compress.add_argument(
        "-e",
        "--extrapolate",
        default=5,
        type=int,
        help="The scale of model extrapolation",
    )
    parser_compress.add_argument(
        "-s",
        "--stride",
        default=0.01,
        type=float,
        help="The uniform stride of tabulation's first table, the second table will "
        "use 10 * stride as it's uniform stride",
    )
    parser_compress.add_argument(
        "-f",
        "--frequency",
        default=-1,
        type=int,
        help="The frequency of tabulation overflow check(If the input environment "
        "matrix overflow the first or second table range). "
        "By default do not check the overflow",
    )
    parser_compress.add_argument(
        "-c",
        "--checkpoint-folder",
        type=str,
        default=".",
        help="path to checkpoint folder",
    )

    # * print docs script **************************************************************
    subparsers.add_parser(
        "doc-train-input",
        parents=[parser_log],
        help="print the documentation (in rst format) of input training parameters.",
    )

    args = parser.parse_args()

    # do not set log handles for None it is useless
    # log handles for train will be set separatelly
    # when the use of MPI will be determined in `RunOptions`
    if args.command not in (None, "train"):
        set_log_handles(args.log_level, Path(args.log_path))

    if args.command is None:
        parser.print_help()
    elif args.command == "train":
        train(args)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "config":
        config(args)
    elif args.command == "test":
        test(args)
    elif args.command == "transform":
        transform(args)
    elif args.command == "compress":
        compress(args)
    elif args.command == "doc-train-input":
        doc_train_input(args)
    else:
        raise RuntimeError(f"unknown command {args.command}")
