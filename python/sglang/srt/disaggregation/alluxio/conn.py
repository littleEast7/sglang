from __future__ import annotations

import time
import logging
import queue
import struct
import threading
from typing import List, Optional
import os

import concurrent
import torch

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import BaseKVSender, KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)

GUARD = "AlluxioMsgGuard".encode("ascii")
FILEPATH = "/tmp/kv_cache"


class AlluxioKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.base_gpu_id = server_args.base_gpu_id
        self.peer_gpu_id = dict()
        self.transfer_queue = queue.Queue()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._start_workers()
        self.status_record = dict()
        self.peer_kv_data_ptrs = dict()
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.transfer_lock = threading.Lock()
        self.decode_kv_indices_table = dict()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()

    def _start_workers(self):
        for _ in range(4):
            self.thread_pool.submit(self._transfer_worker)

    def _transfer_worker(self):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            while True:
                with self.transfer_lock:
                    task = self.transfer_queue.get()
                    if task is None:
                        break
                    # 将关键操作移入锁保护范围
                    self._send_kvcache(task["src_kv_indices"], task["room"])

                    with open(f"{FILEPATH}/{task['room']}_finish_send.bin", "wb") as f:
                        f.write(b"success")  # 修正字节写入
                    # self.status_record[task["room"]] = KVPoll.Success
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            while True:
                for room, status in self.status_record.items():
                    if os.path.exists(f"{FILEPATH}/{room}_finish_send.bin") and status != KVPoll.Success:
                        with self.transfer_lock:
                            self._store_kvcache(self.decode_kv_indices_table[room], room)
                            # self.status_record[room] = KVPoll.Success
                            with open(f"{FILEPATH}/{room}_finish_recv.bin", "wb") as f:
                                f.write(b"success")


    def _start_bootstrap_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                room = int(waiting_req_bytes[2].decode("ascii"))
                if waiting_req_bytes[1] == b"\x00\x00\x00\x00":
                    # self.status_record[int(room)] = KVPoll.WaitingForInput
                    self.peer_gpu_id[room] = int(waiting_req_bytes[4].decode("ascii"))
                    self.peer_kv_data_ptrs[room] = list(struct.unpack(f"{len(waiting_req_bytes[3])//8}Q", waiting_req_bytes[3])),
                    continue
                self.decode_kv_indices_table[room] = np.frombuffer(waiting_req_bytes[1], dtype=np.int32)

                self.status_record[room] = KVPoll.WaitingForInput

        threading.Thread(target=bootstrap_thread).start()

    def _send_kvcache(self, kv_indices: torch.Tensor, room: str):
        """
        使用 CuPy 拷贝 GPU 显存中的 KV Cache 数据，并保存为文件。
        """
        import torch
        import cupy as cp
        import os

        kv_data_ptrs, kv_item_lens = (
            self.kv_args.kv_data_ptrs,
            self.kv_args.kv_item_lens
        )
        num_layers = len(kv_data_ptrs) // 2

        torch.cuda.synchronize()
        os.makedirs(FILEPATH, exist_ok=True)

        import numpy as np
        for layer_id in range(num_layers):
            k_ptr = kv_data_ptrs[layer_id]
            item_len = kv_item_lens[layer_id]
            num_indices = len(kv_indices)

            # 目标 buffer (PyTorch Tensor)
            k_buffer = torch.empty((num_indices, item_len), dtype=torch.uint8, device='cuda')

            for i, index in enumerate(kv_indices):
                src_ptr = k_ptr + int(index) * item_len

                dst_cp = cp.ndarray(
                    (item_len,),
                    dtype=cp.uint8,
                    memptr=cp.cuda.MemoryPointer(
                        cp.cuda.UnownedMemory(
                            k_buffer[i].data_ptr(),
                            item_len,
                            k_buffer[i]  # 持有 PyTorch tensor 的生命周期
                        ),
                        0
                    )
                )
                memptr=cp.cuda.MemoryPointer(
                        cp.cuda.UnownedMemory(
                            src_ptr,
                            item_len,
                            owner=None  # 原始地址没有 Python 对象持有者
                        ),
                        0
                    )
                src_cp = cp.ndarray(
                    (item_len,),
                    dtype=cp.uint8,
                    memptr=memptr
                )

                if not isinstance(src_cp, np.ndarray):
                    src_cp = np.asarray(src_cp.get())
                dst_cp.set(src_cp)

            torch.cuda.synchronize()
            cpu_buffer = k_buffer.cpu().numpy()

            for i in range(len(kv_indices)):
                filename = f"{FILEPATH}/{room}_layer_{layer_id}_num_{i}.bin"
                with open(filename, "wb") as f:
                    cpu_buffer[i].tofile(f)
                print(f"已保存 {item_len / 1024 / 1024:.2f} MB 数据到 {filename}")


    def _store_kvcache(self, dst_kv_indices: torch.Tensor, room: str):
        """
        从文件加载KV Cache数据到目标显存
        """
        import torch
        import cupy as cp
        import ctypes
        # 加载 CUDA Runtime
        libcudart = ctypes.CDLL('libcudart.so')
        # 定义 cudaMemcpy 常量
        cudaMemcpyHostToDevice = 1

        kv_item_lens = self.kv_args.kv_item_lens
        kv_data_ptrs, kv_item_lens = (
            self.kv_args.kv_data_ptrs,
            self.kv_args.kv_item_lens
        )
        num_layers = len(kv_data_ptrs) // 2

        torch.cuda.synchronize()
        # os.makedirs(FILEPATH, exist_ok=True)

        import numpy as np
        for layer_id in range(num_layers):
            k_ptr = kv_data_ptrs[layer_id]
            item_len = kv_item_lens[layer_id]
            # 从文件加载数据到CPU
            cpu_buffer = list()
            for i in range(len(dst_kv_indices)):
                filename = f"{FILEPATH}/{room}_layer_{layer_id}_num_{i}.bin"
                with open(filename, "rb") as f:
                    temp = np.fromfile(f, dtype=np.uint8, count=item_len)
                    cpu_buffer.append(torch.from_numpy(temp))
            # 将数据写入目标显存地址
            for i, dst_index in enumerate(dst_kv_indices):
                dst_ptr = k_ptr + int(dst_index) * item_len
                src_ptr = cpu_buffer[i].data_ptr()
                libcudart.cudaMemcpy(
                    ctypes.c_void_p(dst_ptr),
                    ctypes.c_void_p(src_ptr),
                    ctypes.c_size_t(item_len),
                    ctypes.c_int(cudaMemcpyHostToDevice)
                )

            torch.cuda.synchronize()


class AlluxioKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: AlluxioKVManager,
        bootstrap_room: int,
    ):
        self.mgr = mgr
        self.room = bootstrap_room
        self.mgr.status_record[self.room] = KVPoll.Bootstrapping

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(self, src_kv_indices: torch.Tensor):
        """
        触发KV缓存传输请求
        Args:
            kv_indices: 需要传输的KV缓存索引
            dest_path: 目标存储路径
        """
        # 将请求加入传输队列
        self.mgr.transfer_queue.put({
            "src_kv_indices": src_kv_indices,
            "dst_kv_indices": self.mgr.decode_kv_indices_table[self.room],
            "room": self.room,
            "timestamp": time.time()
        })
        self.mgr.status_record[self.room] = KVPoll.Transferring
        logger.debug(f"Queued transfer request for {len(src_kv_indices)} caches")

    def poll(self) -> KVPoll:
        if os.path.exists(f"{FILEPATH}/{self.room}_finish_recv.bin"):
            self.mgr.status_record[self.room] = KVPoll.Success
        return self.mgr.status_record[self.room]

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class AlluxioKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: AlluxioKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.mgr = mgr
        super().__init__(mgr, bootstrap_addr, bootstrap_room, data_parallel_rank)

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.mgr.kv_args.engine_rank}"
            )
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to {self.prefill_server_url} with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            # 可修改下面部分
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        kv_indices.tobytes() if not is_dummy else b"\x00\x00\x00\x00",
                        str(self.bootstrap_room).encode('ascii'),
                    ]
                )
            self.mgr.decode_kv_indices_table[self.bootstrap_room] = kv_indices
            self.mgr.status_record[self.bootstrap_room] = KVPoll.WaitingForInput

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.mgr.kv_args.aux_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        b"\x00\x00\x00\x00",    # 空
                        str(self.bootstrap_room).encode('ascii'),
                        packed_kv_data_ptrs,
                        str(self.mgr.base_gpu_id).encode('ascii')
                    ]
                )
                self.mgr.status_record[self.bootstrap_room] = KVPoll.WaitingForInput

    def poll(self) -> KVPoll:
        # 通过文件检查是否传输成功
        if os.path.exists(f"{FILEPATH}/{self.bootstrap_room}_finish_recv.bin"):
            self.mgr.status_record[self.bootstrap_room] = KVPoll.Success
        # else:
        #     self.mgr.status_record[self.bootstrap_room] = KVPoll.Failed
        # print("receiver poll, room: ", self.bootstrap_room)
        # print("poll, status: ", self.mgr.status_record[self.bootstrap_room])
        return self.mgr.status_record[self.bootstrap_room]

class AlluxioKVBootstrapServer(CommonKVBootstrapServer):
    pass


