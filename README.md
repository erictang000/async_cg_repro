removing 
```python
 async def test_async(self):
        print(f"role {self.role} rank {self.rank} testing async")
        await asyncio.sleep(2)
        print(f"role {self.role} rank {self.rank} done testing async")
```
in `ray_workers.py` allows the compiled graphs code to work when you run `ray_param_sync.py`. But adding it causes a traceback like the following:

```
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/actor.py", line 1722, in __ray_call__
    return fn(self, *args, **kwargs)
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/experimental/channel/torch_tensor_nccl_channel.py", line 658, in _do_init_communicator
    ctx.communicators[group_id] = _NcclGroup(
  File "/home/ray/anaconda3/lib/python3.10/site-packages/ray/experimental/channel/nccl_group.py", line 90, in __init__
    self._comm = self.nccl_util.NcclCommunicator(world_size, comm_id, rank)
  File "cupy_backends/cuda/libs/nccl.pyx", line 282, in cupy_backends.cuda.libs.nccl.NcclCommunicator.__init__
  File "cupy_backends/cuda/libs/nccl.pyx", line 128, in cupy_backends.cuda.libs.nccl.check_status
cupy_backends.cuda.libs.nccl.NcclError: NCCL_ERROR_INVALID_USAGE: invalid usage (run with NCCL_DEBUG=WARN for details)
```

on the call to `experimental_compile`. This seems to have to do with the `MultiOutputNode` as well.

If in `ray_param_sync.py` we change
```python
# multioutput node + async def in worker breaks!
group_2_outputs = [
    group_2_workers[i].recv_weights.bind(weights)
    for i in range(dp_size_2)
]
sync_param_dag = MultiOutputNode(group_2_outputs)
```
to just receive weights on a single node:
```
# just a single worker works here even with async!!!!
#  sync_param_dag = group_2_workers[0].recv_weights.bind(weights)
```
then the compiled graph compiles even with the async def in the worker.
