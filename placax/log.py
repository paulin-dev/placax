"""Static logging facade over stdlib logging - Log.info("msg") directly, no per-file logger boilerplate."""
import logging
import sys

_FORMAT = "%(message)s"  # plain lines by default - scripts want readable output, not [INFO] noise


def _caller_logger() -> logging.Logger:
    """Returns the stdlib logger for whoever called the Log.* method two frames up."""
    # Walk two stack frames up (past this function and the Log.* method) to
    # find the module that actually made the logging call.
    frame = sys._getframe(2)
    name = frame.f_globals.get("__name__", "placax")
    return logging.getLogger(name)


class Log:
    """Static logging facade so callers embedding placax as a library can control/silence it centrally."""

    @staticmethod
    def debug(msg: str, *args: object) -> None:
        _caller_logger().debug(msg, *args)

    @staticmethod
    def info(msg: str, *args: object) -> None:
        _caller_logger().info(msg, *args)

    @staticmethod
    def warning(msg: str, *args: object) -> None:
        _caller_logger().warning(msg, *args)

    @staticmethod
    def error(msg: str, *args: object) -> None:
        _caller_logger().error(msg, *args)

    @staticmethod
    def configure(level: int = logging.INFO, fmt: str = _FORMAT) -> None:
        """Sets up console logging for the whole process; call once from a script's main(), never from library code."""
        logging.basicConfig(level=level, format=fmt)

    @staticmethod
    def silence(*names: str) -> None:
        """Mutes a logger subtree, defaulting to both "placax" and "placax_agents" if names is empty."""
        for name in names or ("placax", "placax_agents"):
            logging.getLogger(name).setLevel(logging.CRITICAL + 1)
