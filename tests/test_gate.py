import asyncio

import pytest

from app.main import Busy, InferenceGate


def test_gate_rejects_when_queue_is_full():
    async def scenario():
        gate = InferenceGate(concurrency=1, max_queue=1)
        release = asyncio.Event()

        async def hold():
            async with gate.slot():
                await release.wait()

        running = asyncio.create_task(hold())
        await asyncio.sleep(0)
        queued = asyncio.create_task(hold())
        await asyncio.sleep(0)

        with pytest.raises(Busy):
            async with gate.slot():
                pass

        release.set()
        await asyncio.gather(running, queued)
        async with gate.slot():  # capacity is back once the queue drains
            pass

    asyncio.run(scenario())
