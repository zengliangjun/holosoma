"""Core public API."""

from __future__ import annotations

import pathlib
import shutil
import sys
import warnings
from typing import Callable, Literal, Sequence, TypeVar, cast, overload

from typing_extensions import Annotated, assert_never, deprecated

from tyro import (
    _calling,
    _strings,
    _unsafe_cache,
    conf,
)
from tyro import _fmtlib as fmt
from tyro._backends import _argparse as argparse
from tyro._singleton import (
    MISSING_NONPROP,
    NonpropagatingMissingType,
    PropagatingMissingType,
)
from tyro._typing import TypeForm
from tyro.constructors import ConstructorRegistry
from tyro.constructors._primitive_spec import UnsupportedTypeAnnotationError
import tyro

OutT = TypeVar("OutT")


# The overload here is necessary for pyright and pylance due to special-casing
# related to using typing.Type[] as a temporary replacement for
# typing.TypeForm[].
#
# https://github.com/microsoft/pyright/issues/4298


@overload
def cli(
    f: TypeForm[OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: OutT
    | NonpropagatingMissingType
    | PropagatingMissingType = MISSING_NONPROP,
    return_unknown_args: Literal[False] = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    compact_help: bool = False,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
) -> OutT: ...


@overload
def cli(
    f: TypeForm[OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: OutT
    | NonpropagatingMissingType
    | PropagatingMissingType = MISSING_NONPROP,
    return_unknown_args: Literal[True],
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    compact_help: bool = False,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
) -> tuple[OutT, list[str]]: ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    # Passing a default makes sense for things like dataclasses, but are not
    # supported for general callables. These can, however, be specified in the
    # signature of the callable itself.
    default: NonpropagatingMissingType | PropagatingMissingType = MISSING_NONPROP,
    return_unknown_args: Literal[False] = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    compact_help: bool = False,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
) -> OutT: ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    # Passing a default makes sense for things like dataclasses, but are not
    # supported for general callables. These can, however, be specified in the
    # signature of the callable itself.
    default: NonpropagatingMissingType | PropagatingMissingType = MISSING_NONPROP,
    return_unknown_args: Literal[True],
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    compact_help: bool = False,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
) -> tuple[OutT, list[str]]: ...


def cli(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: OutT
    | NonpropagatingMissingType
    | PropagatingMissingType = MISSING_NONPROP,
    return_unknown_args: bool = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    compact_help: bool = False,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    **deprecated_kwargs,
) -> OutT | tuple[OutT, list[str]]:
    """Generate a command-line interface from type annotations and populate the target with arguments.

    :func:`cli()` is the core function of tyro. It takes a type-annotated function or class
    and automatically generates a command-line interface to populate it from user arguments.

    Two main usage patterns are supported:

    1. With a function (CLI arguments become function parameters):

       .. code-block:: python

          import tyro

          def main(a: str, b: str) -> None:
              print(a, b)

          if __name__ == "__main__":
              tyro.cli(main)  # Parses CLI args, calls main() with them

    2. With a class (CLI arguments become object attributes):

       .. code-block:: python

          from dataclasses import dataclass
          from pathlib import Path

          import tyro

          @dataclass
          class Config:
              a: str
              b: str

          if __name__ == "__main__":
              config = tyro.cli(Config)  # Parses CLI args, returns populated AppConfig
              print(f"Config: {config}")

    Args:
        f: The function or type to populate from command-line arguments. This must have
            type-annotated inputs for tyro to work correctly.
        prog: The name of the program to display in the help text. If not specified, the
            script filename is used. This mirrors the argument from
            :py:class:`argparse.ArgumentParser()`.
        description: The description text shown at the top of the help output. If not
            specified, the docstring of `f` is used. This mirrors the argument from
            :py:class:`argparse.ArgumentParser()`.
        args: If provided, parse arguments from this sequence of strings instead of
            the command line. This is useful for testing or programmatic usage. This mirrors
            the argument from :py:meth:`argparse.ArgumentParser.parse_args()`.
        default: An instance to use for default values. This is only supported if ``f`` is a
            type like a dataclass or dictionary, not if ``f`` is a general
            callable like a function. This is useful for merging CLI arguments
            with values loaded from elsewhere, such as a config file. The
            default value is :data:`tyro.MISSING_NONPROP`.
        return_unknown_args: If True, returns a tuple of the output and a list of unknown
            arguments that weren't consumed by the parser. This mirrors the behavior of
            :py:meth:`argparse.ArgumentParser.parse_known_args()`.
        use_underscores: If True, uses underscores as word delimiters in the help text
            instead of hyphens. This only affects the displayed help; both underscores and
            hyphens are treated equivalently during parsing. The default (False) follows the
            GNU style guide for command-line options.
            https://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html
        console_outputs: If set to False, suppresses parsing errors and help messages.
            This is useful in distributed settings where tyro.cli() is called from multiple
            workers but console output is only desired from the main process.
        add_help: Add a -h/--help option to the parser. This mirrors the argument from
            :py:class:`argparse.ArgumentParser()`.
        compact_help: If True, use compact help format that omits full argument descriptions.
            This mode shows only ``--flag TYPE (default: value)`` instead of including
            the full docstring. When enabled, users can access full help with
            ``--help-verbose``. Only applies to the TyroBackend; ignored for ArgparseBackend.
        config: A sequence of configuration marker objects from :mod:`tyro.conf`. This
            allows applying markers globally instead of annotating individual fields.
            For example: ``tyro.cli(Config, config=(tyro.conf.PositionalRequiredArgs,))``
        registry: A :class:`tyro.constructors.ConstructorRegistry` instance containing custom
            constructor rules.

    Returns:
        If ``f`` is a type (like a dataclass), returns an instance of that type populated
        with values from the command line. If ``f`` is a function, calls the function with
        arguments from the command line and returns its result. If ``return_unknown_args``
        is True, returns a tuple of the result and a list of unused command-line arguments.
    """

    # Make sure we start on a clean slate. Some tests may fail without this due to
    # memory address conflicts.
    _unsafe_cache.clear_cache()

    try:
        with _strings.delimeter_context("_" if use_underscores else "-"):
            args = ['exp:g1-29dof-fast-sac', 'simulator:isaacgym', 'logger:disabled', '--training.seed', '1']
            output = tyro._cli._cli_impl(
                f,
                prog=prog,
                description=description,
                args=args,
                default=default,
                return_parser=False,
                return_unknown_args=return_unknown_args,
                use_underscores=use_underscores,
                console_outputs=console_outputs,
                add_help=add_help,
                compact_help=compact_help,
                config=config,
                registry=registry,
                **deprecated_kwargs,
            )
    except UnsupportedTypeAnnotationError as e:
        # Format and display the error nicely.
        error_message = fmt.box["bright_red"](
            fmt.text["bright_red", "bold"]("Invalid input to tyro.cli()"),
            fmt.rows(
                fmt.text("Could not create CLI parser from the provided type."),
                fmt.hr["red"](),
                *[fmt.cols((fmt.text["dim"]("• "), 2), msg) for msg in e.message],
            ),
        )
        print(
            "\n".join(
                error_message.render(width=min(shutil.get_terminal_size()[0], 80))
            ),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    # Prevent unnecessary memory usage.
    _unsafe_cache.clear_cache()

    if return_unknown_args:
        assert isinstance(output, tuple)
        run_with_args_from_cli = output[0]
        out = run_with_args_from_cli()
        while isinstance(out, _calling.DummyWrapper):
            out = out.__tyro_dummy_inner__
        return out, output[1]  # type: ignore
    else:
        run_with_args_from_cli = cast(Callable[[], OutT], output)
        out = run_with_args_from_cli()
        while isinstance(out, _calling.DummyWrapper):
            out = out.__tyro_dummy_inner__
        return out

