import gi
import logging

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib  # type: ignore

logger = logging.getLogger(__name__)


def get_queue(index: int = 0, args: dict = None):
    """
    return a queue element with the given index and args
        - index: the index of the queue element, used to set the name of the element
    """

    queue = Gst.ElementFactory.make("queue", f"queue-{index:02}")
    return queue
