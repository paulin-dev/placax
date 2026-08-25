"""Static logging facade over stdlib logging - Log.info("msg") directly,
no per-file `logger = logging.getLogger(__name__)` boilerplate. Each call
is routed to a logger named after its caller's module (via a stack
lookup), so per-module silence() still works."""
import logging
import sys

_FORMAT = "%(message)s"  # plain lines by default - scripts want readable output, not [INFO] noise


def _caller_logger() -> logging.Logger:
    """The stdlib logger for whoever called the Log.* method two frames up."""
    frame = sys._getframe(2)  # 0=here, 1=the Log.* method, 2=the actual caller
    name = frame.f_globals.get("__name__", "placax")
    return logging.getLogger(name)


class Log:
    """Static logging facade used across the library and scripts, so a
    caller embedding this as a library can control or silence it via
    configure()/silence()."""

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
        """Sets up console logging for the whole process. Call once, e.g.
        at the top of a script's main(); library code should never call
        this itself."""
        logging.basicConfig(level=level, format=fmt)

    @staticmethod
    def silence(*names: str) -> None:
        """Mutes a logger subtree. Defaults to "placax" and "placax_agents"
        (separate top-level names, so both are silenced unless names
        overrides them)."""
        for name in names or ("placax", "placax_agents"):
            logging.getLogger(name).setLevel(logging.CRITICAL + 1)
