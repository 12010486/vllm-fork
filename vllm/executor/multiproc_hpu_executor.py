import asyncio
import os
from functools import partial
from typing import Any, List, Optional

import torch

from vllm.executor.distributed_gpu_executor import (# yapf: disable
    DistributedGPUExecutor, DistributedGPUExecutorAsync)
from vllm.executor.hpu_executor import create_worker  
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.triton_utils.importing import HAS_TRITON
from vllm.utils import (_run_task_with_lock, cuda_device_count_stateless,
                        cuda_is_initialized, get_distributed_init_method,
                        get_open_port, get_vllm_instance_id, make_async,
                        update_environment_variables)

if HAS_TRITON:
    from vllm.triton_utils import maybe_set_triton_cache_manager

logger = init_logger(__name__)


class MultiprocessingHPUExecutor(DistributedGPUExecutor):  
    """Python multiprocessing-based multi-HPU executor"""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        self._check_executor_parameters()

        # Create the parallel HPU workers.
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Configure thread parallelism if OMP_NUM_THREADS isn't set
        default_omp_num_threads = 1
        if "OMP_NUM_THREADS" not in os.environ and (
                current_parallelism := torch.get_num_threads()) > default_omp_num_threads:
            logger.warning(
                "Reducing Torch parallelism from %d threads to %d to avoid "
                "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
                "external environment to tune this value as needed.",
                current_parallelism, default_omp_num_threads)
            os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
            torch.set_num_threads(default_omp_num_threads)

        # workaround for https://github.com/vllm-project/vllm/issues/6103
        if HAS_TRITON and world_size > 1:
            maybe_set_triton_cache_manager()

        # Multiprocessing-based executor does not support multi-node setting.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        self.workers: List[ProcessWorkerWrapper] = []
        self.tp_driver_workers: List[ProcessWorkerWrapper] = []
        self.non_driver_workers: List[ProcessWorkerWrapper] = []

        if world_size == 1:
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            for rank in range(1, world_size):
                worker = ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        create_worker,
                        **self._get_create_worker_kwargs(
                            rank=rank,
                            local_rank=rank,
                            distributed_init_method=distributed_init_method,
                        )))
                self.workers.append(worker)
                if rank % tensor_parallel_size == 0:
                    self.tp_driver_workers.append(worker)
                else:
                    self.non_driver_workers.append(worker)

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_worker(
            distributed_init_method=distributed_init_method)
        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

    def _check_executor_parameters(self):
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            update_environment_variables({
                "CUDA_VISIBLE_DEVICES": (",".join(map(str, range(world_size))))
            })

        if (cuda_is_initialized()
                and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"):
            logger.warning("CUDA was previously initialized. We must use "
                           "the `spawn` multiprocessing start method. Setting "
                           "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.")
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cuda_device_count = cuda_device_count_stateless()
        assert tensor_parallel_size <= cuda_device_count, (
            f"please set tensor_parallel_size ({tensor_parallel_size}) "
            f"to less than max local hpu count ({hpu_device_count})")

        assert world_size <= cuda_device_count, (
            f"please ensure that world_size ({world_size}) "
            f"is less than than max local hpu count ({hpu_device_count})")

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

    def _driver_execute_model(
        self, execute_model_req: Optional[ExecuteModelRequest]
    ) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(execute_model_req)

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        if async_run_tensor_parallel_workers_only:
            return [
                worker.execute_method(method, *args, **kwargs)
                for worker in self.non_driver_workers
            ]

        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and not self.worker_monitor.is_alive(
        ):
            raise RuntimeError("Worker processes are not running")

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()


class MultiprocessingHPUExecutorAsync(MultiprocessingHPUExecutor,
                                      DistributedGPUExecutorAsync):  

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
        self.pp_locks: Optional[List[asyncio.Lock]] = None

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if not self.tp_driver_workers:
            return await self.driver_exec_model(execute_model_req)

        if self.pp_locks is None:
            self.pp_locks = [
                asyncio.Lock()
                for _ in range(self.parallel_config.pipeline_parallel_size)
            ]

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(self.driver_exec_model, self.pp_locks[0],
                                    execute_model_req))
        ]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers,
                                                start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(driver_worker.execute_method_async,
                                        self.pp_locks[pp_rank],
                                        "execute_model", execute_model_req)))
        results = await asyncio.gather(*tasks)

        return results[-1]

    async def _start_worker_execution_loop(self):
        coros = [
            worker.execute_method_async("start_worker_execution_loop")
            for worker in self.non_driver_workers
        ]
        return await asyncio.gather(*coros)
