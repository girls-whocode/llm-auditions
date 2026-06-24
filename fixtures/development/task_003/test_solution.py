import threading
import time
from solution import RateLimiter


def test_rate_limiter_bounds():
    limiter = RateLimiter(max_calls=2, period_seconds=1)
    start = time.time()
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    elapsed = time.time() - start
    assert elapsed >= 0.9
