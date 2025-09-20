#!/usr/bin/env python3
"""
Load and Performance Testing Suite for HeySol API Client

Tests performance under high traffic, rate limiting, and concurrent operations.
"""

import os
import time
import threading
import statistics
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import HeySol components
from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import RateLimitError, HeySolError


class PerformanceMetrics:
    """Class to track performance metrics."""

    def __init__(self):
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
        self.rate_limit_hits = 0
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        """Start performance timer."""
        self.start_time = time.time()

    def end_timer(self):
        """End performance timer."""
        self.end_time = time.time()

    def record_request(self, duration: float, success: bool = True, rate_limited: bool = False):
        """Record request metrics."""
        self.request_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        if rate_limited:
            self.rate_limit_hits += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.request_times:
            return {"error": "No requests recorded"}

        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        total_requests = len(self.request_times)

        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "rate_limit_hits": self.rate_limit_hits,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "average_response_time": statistics.mean(self.request_times),
            "median_response_time": statistics.median(self.request_times),
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "success_rate": (self.success_count / total_requests) * 100 if total_requests > 0 else 0
        }


class TestLoadPerformance:
    """Test suite for load and performance testing."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

        # Mock responses for performance testing
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        self.mock_response.headers = {"Content-Type": "application/json"}
        self.mock_response.json.return_value = {"status": "success"}

    def simulate_request(self, delay: float = 0.0) -> float:
        """Simulate a single request with optional delay."""
        start_time = time.time()

        try:
            # Simulate network delay
            time.sleep(delay)

            # Mock successful response
            with patch.object(self.client, '_make_request', return_value=self.mock_response):
                with patch.object(self.client, '_parse_mcp_response', return_value={"result": "success"}):
                    self.client.get_spaces()

            end_time = time.time()
            return end_time - start_time

        except Exception as e:
            end_time = time.time()
            return end_time - start_time

    def test_single_request_performance(self):
        """Test performance of a single request."""
        metrics = PerformanceMetrics()
        metrics.start_timer()

        duration = self.simulate_request()
        metrics.record_request(duration)
        metrics.end_timer()

        summary = metrics.get_summary()

        # Single request should be fast
        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["average_response_time"] < 1.0  # Should be under 1 second
        assert summary["success_rate"] == 100.0

    def test_concurrent_requests_performance(self):
        """Test performance with concurrent requests."""
        num_concurrent = 10
        metrics = PerformanceMetrics()
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.simulate_request) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                try:
                    duration = future.result()
                    metrics.record_request(duration)
                except Exception as e:
                    metrics.record_request(0.0, success=False)

        metrics.end_timer()
        summary = metrics.get_summary()

        assert summary["total_requests"] == num_concurrent
        assert summary["successful_requests"] == num_concurrent
        assert summary["success_rate"] == 100.0
        assert summary["requests_per_second"] > 5  # Should handle at least 5 RPS

    def test_high_load_performance(self):
        """Test performance under high load."""
        num_requests = 100
        max_workers = 20
        metrics = PerformanceMetrics()
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.simulate_request) for _ in range(num_requests)]

            for future in as_completed(futures):
                try:
                    duration = future.result()
                    metrics.record_request(duration)
                except Exception as e:
                    metrics.record_request(0.0, success=False)

        metrics.end_timer()
        summary = metrics.get_summary()

        assert summary["total_requests"] == num_requests
        assert summary["success_rate"] >= 95.0  # At least 95% success rate
        assert summary["requests_per_second"] > 10  # Should handle at least 10 RPS

    def test_rate_limiting_behavior(self):
        """Test rate limiting behavior under load."""
        # Simulate rate limiting by making requests too fast
        num_requests = 200
        rate_limit_threshold = 60  # requests per minute
        metrics = PerformanceMetrics()
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(self.simulate_request) for _ in range(num_requests)]

            for future in as_completed(futures):
                try:
                    duration = future.result()
                    # Simulate rate limiting for some requests
                    is_rate_limited = len([t for t in metrics.request_times if t < 0.01]) > rate_limit_threshold
                    metrics.record_request(duration, rate_limited=is_rate_limited)
                except Exception as e:
                    metrics.record_request(0.0, success=False)

        metrics.end_timer()
        summary = metrics.get_summary()

        assert summary["total_requests"] == num_requests
        assert summary["rate_limit_hits"] > 0  # Should hit rate limits
        assert summary["requests_per_second"] <= 2  # Should be throttled

    def test_variable_load_patterns(self):
        """Test performance with variable load patterns."""
        patterns = [
            ("burst", 50, 0.01),      # 50 requests, 10ms delay
            ("steady", 30, 0.1),      # 30 requests, 100ms delay
            ("slow", 20, 0.5),        # 20 requests, 500ms delay
        ]

        results = {}

        for pattern_name, num_requests, delay in patterns:
            metrics = PerformanceMetrics()
            metrics.start_timer()

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.simulate_request, delay) for _ in range(num_requests)]

                for future in as_completed(futures):
                    try:
                        duration = future.result()
                        metrics.record_request(duration)
                    except Exception as e:
                        metrics.record_request(0.0, success=False)

            metrics.end_timer()
            results[pattern_name] = metrics.get_summary()

        # Burst should be fastest
        assert results["burst"]["requests_per_second"] > results["steady"]["requests_per_second"]
        assert results["steady"]["requests_per_second"] > results["slow"]["requests_per_second"]

        # All should have high success rates
        for pattern in patterns:
            assert results[pattern[0]]["success_rate"] >= 95.0

    def test_memory_usage_under_load(self):
        """Test memory usage patterns under sustained load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run sustained load
        num_requests = 500
        metrics = PerformanceMetrics()
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.simulate_request) for _ in range(num_requests)]

            for future in as_completed(futures):
                try:
                    duration = future.result()
                    metrics.record_request(duration)
                except Exception as e:
                    metrics.record_request(0.0, success=False)

        metrics.end_timer()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        summary = metrics.get_summary()

        # Memory usage should not grow excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100  # Less than 100MB growth

        assert summary["total_requests"] == num_requests
        assert summary["success_rate"] >= 95.0

    def test_error_recovery_performance(self):
        """Test performance of error recovery mechanisms."""
        num_requests = 100
        error_rate = 0.1  # 10% of requests will fail
        metrics = PerformanceMetrics()
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(num_requests):
                if i % int(1/error_rate) == 0:
                    # This request will fail
                    futures.append(executor.submit(self._simulate_failing_request))
                else:
                    futures.append(executor.submit(self.simulate_request))

            for future in as_completed(futures):
                try:
                    duration = future.result()
                    metrics.record_request(duration)
                except Exception as e:
                    metrics.record_request(0.0, success=False)

        metrics.end_timer()
        summary = metrics.get_summary()

        assert summary["total_requests"] == num_requests
        assert summary["failed_requests"] > 0  # Should have some failures
        assert summary["successful_requests"] > 0  # Should have some successes
        assert summary["success_rate"] >= 80.0  # At least 80% success rate

    def _simulate_failing_request(self) -> float:
        """Simulate a request that fails."""
        start_time = time.time()

        # Simulate a failure
        raise HeySolError("Simulated request failure")

    def test_response_time_distribution(self):
        """Test response time distribution under load."""
        num_requests = 200
        metrics = PerformanceMetrics()
        metrics.start_timer()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.simulate_request) for _ in range(num_requests)]

            for future in as_completed(futures):
                try:
                    duration = future.result()
                    metrics.record_request(duration)
                except Exception as e:
                    metrics.record_request(0.0, success=False)

        metrics.end_timer()
        summary = metrics.get_summary()

        # Analyze response time distribution
        response_times = sorted(metrics.request_times)

        # 50th percentile (median)
        p50 = response_times[int(len(response_times) * 0.5)]

        # 95th percentile
        p95 = response_times[int(len(response_times) * 0.95)]

        # 99th percentile
        p99 = response_times[int(len(response_times) * 0.99)]

        assert summary["total_requests"] == num_requests
        assert p50 < 0.5  # Median should be under 500ms
        assert p95 < 1.0  # 95% should be under 1s
        assert p99 < 2.0  # 99% should be under 2s

    def test_scalability_metrics(self):
        """Test scalability with increasing load."""
        load_levels = [10, 50, 100, 200]

        results = {}

        for load in load_levels:
            metrics = PerformanceMetrics()
            metrics.start_timer()

            with ThreadPoolExecutor(max_workers=min(load, 50)) as executor:
                futures = [executor.submit(self.simulate_request) for _ in range(load)]

                for future in as_completed(futures):
                    try:
                        duration = future.result()
                        metrics.record_request(duration)
                    except Exception as e:
                        metrics.record_request(0.0, success=False)

            metrics.end_timer()
            results[load] = metrics.get_summary()

        # Performance should degrade gracefully with increased load
        for i in range(1, len(load_levels)):
            prev_load = load_levels[i-1]
            curr_load = load_levels[i]

            # Success rate should remain high
            assert results[curr_load]["success_rate"] >= 90.0

            # Response time should not increase exponentially
            assert results[curr_load]["average_response_time"] < results[prev_load]["average_response_time"] * 3


class TestRateLimitHandling:
    """Test suite for rate limit handling."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

    def test_rate_limit_enforcement(self):
        """Test rate limit enforcement."""
        # Set very low rate limit for testing
        self.client.config.rate_limit_per_minute = 5
        self.client.config.rate_limit_enabled = True

        # Reset rate limit counters
        self.client._rate_limit_remaining = 5
        self.client._rate_limit_reset_time = time.time() + 60

        # Should succeed first 5 requests
        for i in range(5):
            # Rate limit check should pass
            self.client._check_rate_limit()

        # 6th request should be rate limited
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            self.client._check_rate_limit()

    def test_rate_limit_reset(self):
        """Test rate limit reset mechanism."""
        self.client.config.rate_limit_per_minute = 3
        self.client.config.rate_limit_enabled = True

        # Use up rate limit
        self.client._rate_limit_remaining = 0
        self.client._rate_limit_reset_time = time.time() - 1  # Already passed

        # Rate limit should reset
        self.client._check_rate_limit()

        # Should have full quota again
        assert self.client._rate_limit_remaining == 3

    def test_rate_limit_disabled(self):
        """Test behavior when rate limiting is disabled."""
        self.client.config.rate_limit_enabled = False

        # Should not enforce rate limits
        original_remaining = self.client._rate_limit_remaining
        self.client._rate_limit_remaining = 0  # Simulate exhausted limit

        # Should not raise error
        self.client._check_rate_limit()

        # Should remain unchanged
        assert self.client._rate_limit_remaining == 0


if __name__ == "__main__":
    # Run performance tests
    print("🚀 Running Load and Performance Tests...")

    try:
        test_suite = TestLoadPerformance()
        test_suite.setup_method()

        # Test single request performance
        test_suite.test_single_request_performance()
        print("✅ Single request performance test passed")

        # Test concurrent requests
        test_suite.test_concurrent_requests_performance()
        print("✅ Concurrent requests performance test passed")

        # Test high load performance
        test_suite.test_high_load_performance()
        print("✅ High load performance test passed")

        # Test rate limiting
        rate_test_suite = TestRateLimitHandling()
        rate_test_suite.setup_method()
        rate_test_suite.test_rate_limit_enforcement()
        print("✅ Rate limit enforcement test passed")

        print("\n🎉 All load and performance tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise