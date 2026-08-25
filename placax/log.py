"""Static logging facade over stdlib logging - call Log.info("msg")
directly, no per-file `logger = logging.getLogger(__name__)` boilerplate.
Each call is still routed to a logger named after its CALLER's module
(via a stack lookup, the same technique stdlib logging itself uses
internally to find caller info), so per-module level control - the
reason that boilerplate exists in the first place - still works via
silence()."""
import logging
import sys

_FORMAT = "%(message)s"  # plain lines by default - scripts want readable output, not [INFO] noise


def _caller_logger() -> logging.Logger:
    """The stdlib logger for whoever called the Log.* method two frames up."""
    frame = sys._getframe(2)  # 0=here, 1=the Log.* method, 2=the actual caller
    name = frame.f_globals.get("__name__", "placax")
    return logging.getLogger(name)


class Log:
    """Static logging facade used across the library and scripts.
    Library code logs through this instead of calling print() directly,
    so a caller using this as a library (not running a script) can
    control or silence it - configure()/silence() change behavior for
    the whole process, not just one call site."""

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
        at the top of a script's main() - library code should never call
        this itself, only Log.info()/.warning()/etc."""
        logging.basicConfig(level=level, format=fmt)

    @staticmethod
    def silence(*names: str) -> None:
        """Mutes a logger subtree. Defaults to every module path this
        library logs under - "placax" and "placax_agents" are separate
        top-level names, not one hierarchy, so both are silenced unless
        names overrides them. Useful when embedding this as a library."""
        for name in names or ("placax", "placax_agents"):
            logging.getLogger(name).setLevel(logging.CRITICAL + 1)
