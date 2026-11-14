import logging
import time

logger = logging.getLogger(__name__)

_RETRYABLE = (TimeoutError, ConnectionError)


def with_retries(fn, *, attempts: int = 3, backoff_s: float = 1.0):
    """Bounded retry for transient provider errors.

    Retries ``fn`` up to ``attempts`` times with exponential backoff for:
    builtin Timeout/ConnectionError, and provider-SDK exceptions carrying
    a transient HTTP ``status_code`` (429/5xx). Anything else propagates
    immediately — auth/config errors shouldn't be retried.

    Author: venkateshwara doijode
    """
    for i in range(attempts):
        try:
            return fn()
        except _RETRYABLE:
            if i == attempts - 1:
                raise
            time.sleep(backoff_s * (2**i))
        except Exception as exc:  # provider SDK errors that look transient
            transient = getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)
            if not transient or i == attempts - 1:
                raise
            logger.warning("transient LLM error %s, retrying", exc)
            time.sleep(backoff_s * (2**i))
