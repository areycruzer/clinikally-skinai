import asyncio
import logging
from typing import Set

class QueueLogHandler(logging.Handler):
    """Custom logging Handler that puts log messages into active asyncio Queues."""
    def __init__(self):
        super().__init__()
        self.queues: Set[asyncio.Queue] = set()

    def emit(self, record):
        try:
            msg = self.format(record)
            for q in list(self.queues):
                try:
                    loop = asyncio.get_running_loop()
                    # Thread-safe and queue-full safe insertion
                    def safe_put(queue, item):
                        if queue.full():
                            try:
                                queue.get_nowait()
                            except Exception:
                                pass
                        try:
                            queue.put_nowait(item)
                        except Exception:
                            pass
                    loop.call_soon_threadsafe(safe_put, q, msg)
                except RuntimeError:
                    # No event loop running in this thread
                    pass
                except Exception:
                    pass
        except Exception:
            pass

# Singleton logger stream handler
logs_stream_handler = QueueLogHandler()
logs_stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))

# Register it globally on the root logger
logging.getLogger().addHandler(logs_stream_handler)
