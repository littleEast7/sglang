import os
import torch
import time
from threading import RLock
from typing import Tuple, Dict
from sglang.srt.metrics.collector import DiskKVCacheMetrics

GB = 1024 * 1024 * 1024
class DiskKVCache:
    """
    Thread-safe disk-based KV cache using memory-mapped files
    Implements persistent storage for KV cache with layer-level locking
    """

    def __init__(self,
                 size: int,
                 page_size: int,
                 dtype: torch.dtype,
                 head_num: int,
                 head_dim: int,
                 layer_num: int,
                 max_capacity_gb: int = 10,
                 cache_dir: str = "./kv_cache",
                 metrics: DiskKVCacheMetrics = None):
        """
        Initialize disk cache with memory-mapped files

        Args:
            size: Total cache capacity in tokens
            page_size: Size of each memory page
            dtype: Data type for storage (e.g., torch.float16)
            head_num: Number of attention heads
            head_dim: Dimension of each attention head
            layer_num: Number of transformer layers
            max_capacity_gb: The max capacity of disk cache
            cache_dir: Directory for cache files
            metrics: Metrics for disk cache
        """
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cache_dir = cache_dir
        self._file_locks: Dict[int, RLock] = {}  # Per-layer file locks
        self.max_capacity_gb = max_capacity_gb
        self.available_layers = set()

        # check the capacity
        self._check_capacity()

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize memory-mapped buffers
        self.k_buffers = []
        self.v_buffers = []
        for i in range(layer_num):
            self._file_locks[i] = RLock()
            self._initialize_layer_file(i)

        self.metrics = metrics
        self._hits = 0
        self._misses = 0
        self._total_elements = (self.size + self.page_size) * self.head_num * self.head_dim
        self._element_size = torch.tensor([], dtype=self.dtype).element_size()

        if self.metrics:
            total_size = self._total_elements * self._element_size * self.layer_num * 2
            self.metrics.cache_capacity_bytes.labels(**self.metrics.labels).set(total_size)
            self.metrics.cache_usage_bytes.labels(**self.metrics.labels).set(0)

    def _check_capacity(self):
        """Check if the maximum capacity limit is exceeded"""
        if self.max_capacity_gb is None:
            return

        total_elements = (self.size + self.page_size) * self.head_num * self.head_dim
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_size_gb = (total_elements * element_size * self.layer_num * 2) / GB

        if total_size_gb > self.max_capacity_gb:
            raise ValueError(
                f"Requested cache size {total_size_gb:.2f}GB "
                f"exceeds maximum capacity {self.max_capacity_gb}GB"
            )

    def _initialize_layer_file(self, layer_id: int):
        """Create and initialize memory-mapped file for single layer"""
        file_path = os.path.join(self.cache_dir, f"layer_{layer_id}.bin")

        # Calculate required file size
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        file_size = (self.size + self.page_size) * self.head_num * self.head_dim * element_size * 2

        # Pre-allocate file space
        with open(file_path, "wb") as f:
            f.truncate(file_size)

        # Create memory-mapped tensors
        cache_buffer = torch.from_file(
            file_path,
            dtype=self.dtype,
            size=(self.size + self.page_size) * self.head_num * self.head_dim * 2,
            shared=True  # Enable multi-process sharing
        ).view(2, self.size + self.page_size, self.head_num, self.head_dim)

        self.k_buffers.append(cache_buffer[0])
        self.v_buffers.append(cache_buffer[1])

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV buffers for specified layer with lock protection

        Args:
            layer_id: Layer index to retrieve

        Returns:
            Tuple of (key_buffer, value_buffer) tensors
        """
        with self._file_locks[layer_id]:
            if layer_id not in self.available_layers:
                raise RuntimeError(f"layer {layer_id}'s kvcache is not set")
            k_cache = self.k_buffers[layer_id]
            v_cache = self.v_buffers[layer_id]
            return k_cache, v_cache

    def get_kv_buffer_with_metrics(self, layer_id: int, get_option_total_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV buffers for specified layer with lock protection

        Args:
            layer_id: Layer index to retrieve
            get_option_total_num: The total number of get options

        Returns:
            Tuple of (key_buffer, value_buffer) tensors
        """
        start_time = time.time()
        with self._file_locks[layer_id]:
            if layer_id not in self.available_layers:
                if self.metrics:
                    self._misses += 1
                    self._update_hit_rate(get_option_totel_num)
                raise RuntimeError(f"layer {layer_id}'s kvcache is not set")

            # Calculate data size being read
            k_size = self.k_buffers[layer_id].numel() * self._element_size
            v_size = self.v_buffers[layer_id].numel() * self._element_size
            total_bytes = k_size + v_size
            k_cache = self.k_buffers[layer_id]
            v_cache = self.v_buffers[layer_id]
            if self.metrics:
                self._hits += 1
                self._update_hit_rate(get_option_totel_num)
                self.metrics.read_ops.labels(**self.metrics.labels).inc()

                # Calculate and record read speed
                duration = time.time() - start_time
                if duration > 0:
                    speed = total_bytes / duration
                    self.metrics.read_speed.labels(**self.metrics.labels).observe(speed)

            return k_cache, v_cache

    def set_kv_buffer(self, layer_id: int, loc: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        start_time = time.time()
        with self._file_locks[layer_id]:
            self.available_layers.add(layer_id)
            self.k_buffers[layer_id][loc] = k_cache.cpu()
            self.v_buffers[layer_id][loc] = v_cache.cpu()

        if self.metrics:
            # Calculate data size being written
            k_size = k_cache.numel() * self._element_size
            v_size = v_cache.numel() * self._element_size
            total_bytes = k_size + v_size

            # Update metrics
            self.metrics.write_ops.labels(**self.metrics.labels).inc()

            # Calculate and record write speed
            duration = time.time() - start_time
            if duration > 0:
                speed = total_bytes / duration
                self.metrics.write_speed.labels(**self.metrics.labels).observe(speed)

            # Update usage metrics
            added_size = k_size + v_size
            current_usage = self.metrics.cache_usage_bytes.labels(**self.metrics.labels)._value.get()
            self.metrics.cache_usage_bytes.labels(**self.metrics.labels).set(current_usage + added_size)

            layer_usage = self.k_buffers[layer_id].numel() * self._element_size + \
                          self.v_buffers[layer_id].numel() * self._element_size
            self.metrics.layer_usage.labels(layer_id=str(layer_id), **self.metrics.labels).set(layer_usage)

    def cleanup(self):
        """Clean up all cache resources"""
        for lock in self._file_locks.values():
            with lock:  # Ensure all operations complete
                pass
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
        except Exception as e:
            print(f"Cache cleanup error: {str(e)}")

    def _update_hit_rate(self, option_total_num):
        """Update the cache hit rate metric"""
        if not self.metrics:
            return
        total = option_total_num
        hit_rate = self._hits / total if total > 0 else 0.0
        self.metrics.cache_hit_rate.labels(**self.metrics.labels).set(hit_rate)

    def get_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def update_metrics(self):
        """Update all monitoring metrics"""
        if not self.metrics:
            return

        # Calculate and update total cache usage
        total_usage = 0
        for layer_id in range(self.layer_num):
            if layer_id in self.available_layers:
                # Calculate layer usage (keys + values)
                layer_usage = self.k_buffers[layer_id].numel() * self._element_size + \
                              self.v_buffers[layer_id].numel() * self._element_size
                total_usage += layer_usage
                # Update per-layer usage metric
                self.metrics.layer_usage.labels(layer_id=str(layer_id), **self.metrics.labels).set(layer_usage)

        # Update total cache usage metric
        self.metrics.cache_usage_bytes.labels(**self.metrics.labels).set(total_usage)
