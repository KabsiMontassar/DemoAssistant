"""
Performance Monitoring Module
Tracks latencies and metrics for RAG pipeline components.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetric:
    """Container for latency measurements."""
    component: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Monitors performance metrics for RAG pipeline components.
    Tracks latencies, throughput, and errors.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent measurements to keep for rolling stats
        """
        self.window_size = window_size
        
        # Store recent latencies per component
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Error tracking
        self._errors: Dict[str, int] = defaultdict(int)
        
        # Request tracking
        self._total_requests = 0
        self._start_time = time.time()
        
        logger.info(f"Performance monitor initialized (window: {window_size})")
    
    def record_latency(self, component: str, duration_ms: float, metadata: Optional[Dict] = None):
        """
        Record latency for a component.
        
        Args:
            component: Component name
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        metric = LatencyMetric(
            component=component,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        self._latencies[component].append(metric)
        
        # Log if slow
        if duration_ms > 1000:  # Slower than 1 second
            logger.warning(
                f"Slow component: {component} took {duration_ms:.1f}ms "
                f"{metadata or ''}"
            )
    
    def record_error(self, component: str):
        """Record an error for a component."""
        self._errors[component] += 1
    
    def record_request(self):
        """Record a new request."""
        self._total_requests += 1
    
    def get_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            component: Specific component (None = all components)
            
        Returns:
            Dict with statistics
        """
        if component:
            return self._get_component_stats(component)
        
        # Get stats for all components
        stats = {
            "total_requests": self._total_requests,
            "uptime_seconds": time.time() - self._start_time,
            "components": {}
        }
        
        for comp in self._latencies.keys():
            stats["components"][comp] = self._get_component_stats(comp)
        
        return stats
    
    def _get_component_stats(self, component: str) -> Dict[str, Any]:
        """Get statistics for a specific component."""
        latencies = self._latencies.get(component, deque())
        
        if not latencies:
            return {
                "calls": 0,
                "errors": self._errors.get(component, 0)
            }
        
        durations = [m.duration_ms for m in latencies]
        
        return {
            "calls": len(latencies),
            "errors": self._errors.get(component, 0),
            "latency_ms": {
                "min": min(durations),
                "max": max(durations),
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "p95": self._percentile(durations, 95),
                "p99": self._percentile(durations, 99)
            }
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
    
    def get_pipeline_breakdown(self) -> Dict[str, float]:
        """
        Get average latency breakdown for the entire pipeline.
        
        Returns:
            Dict mapping component -> average latency (ms)
        """
        breakdown = {}
        for component, latencies in self._latencies.items():
            if latencies:
                durations = [m.duration_ms for m in latencies]
                breakdown[component] = statistics.mean(durations)
        
        return breakdown
    
    def log_summary(self):
        """Log a performance summary."""
        stats = self.get_stats()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total requests: {stats['total_requests']}")
        logger.info(f"Uptime: {stats['uptime_seconds']:.1f}s")
        
        if stats['total_requests'] > 0:
            logger.info(
                f"Throughput: {stats['total_requests'] / stats['uptime_seconds']:.2f} req/s"
            )
        
        breakdown = self.get_pipeline_breakdown()
        if breakdown:
            logger.info("\nComponent Latencies (avg):")
            for component, latency in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {component:30s}: {latency:8.1f}ms")
        
        # Show errors if any
        if self._errors:
            logger.info("\nErrors:")
            for component, count in self._errors.items():
                logger.info(f"  {component:30s}: {count}")
        
        logger.info("=" * 60)
    
    def reset(self):
        """Reset all metrics."""
        self._latencies.clear()
        self._errors.clear()
        self._total_requests = 0
        self._start_time = time.time()
        logger.info("Performance metrics reset")


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, monitor: PerformanceMonitor, component: str, metadata: Optional[Dict] = None):
        """
        Initialize timer.
        
        Args:
            monitor: PerformanceMonitor instance
            component: Component name
            metadata: Optional metadata
        """
        self.monitor = monitor
        self.component = component
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record latency."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        if exc_type is not None:
            # Error occurred
            self.monitor.record_error(self.component)
        
        self.monitor.record_latency(self.component, duration_ms, self.metadata)
        
        return False  # Don't suppress exceptions
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


# Global monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def set_monitor(monitor: PerformanceMonitor):
    """Set the global performance monitor."""
    global _monitor
    _monitor = monitor


def timed(component: str, metadata: Optional[Dict] = None):
    """
    Decorator for timing functions.
    
    Usage:
        @timed("my_component")
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with Timer(monitor, component, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator
