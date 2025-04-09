import os
import sys
import time
import enum
import atexit
import traceback
from functools import partial as bind
from typing import Dict, Callable, Tuple, Any, List

from racecar_env import Racecar


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive: Callable, callid: int):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete: bool = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):
        import multiprocessing
        import cloudpickle

        self._context = multiprocessing.get_context("spawn")
        self._pipe, pipe = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(
            target=self._loop, args=(pipe, fn, initializers), daemon=daemon
        )
        self._process.start()
        self._nextid: int = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs) -> Future:
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self) -> None:
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, IOError):
            pass  # The connection was already closed.

        try:
            self._process.join(0.1)
            if self._process.exitcode is None:
                try:
                    os.kill(self._process.pid, 9)
                    time.sleep(0.1)
                except Exception:
                    pass

        except (AttributeError, AssertionError):
            pass

    def _submit(
        self,
        message: Message,
        payload: Tuple[tuple, Dict[str, Any]] | None=None
    ) -> Future:
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid: int):
        while callid not in self._results:
            try:
                message: Message
                payload: Tuple[tuple, Dict[str, Any]]
                message, callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")

            if message == Message.ERROR:
                raise Exception(payload)

            assert message == Message.RESULT, message
            self._results[callid] = payload

        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function_bytes: bytes, initializers_bytes: bytes) -> None:
        try:
            callid: int | None = None
            state = None
            import cloudpickle

            initializers: List[Callable] = cloudpickle.loads(initializers_bytes)
            function: Callable = cloudpickle.loads(function_bytes)
            [fn() for fn in initializers]

            while True:
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                
                message: Message
                payload: Tuple[tuple, Dict[str, Any]]
                message, callid, payload = pipe.recv()

                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))

                elif message == Message.STOP:
                    return

                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))

                else:
                    raise KeyError(f"Invalid message: {message}")

        except (EOFError, KeyboardInterrupt):
            return

        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
            return

        finally:
            try:
                pipe.close()
            except Exception:
                pass


class Worker:
    initializers = []

    def __init__(self, fn, strategy="thread", state: bool=False):
        if not state:
            def fn(s, *args, fn=fn, **kwargs):
                return (s, fn(*args, **kwargs))

        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise: Future | None = None

    def __call__(self, *args, **kwargs) -> Future:
        self.promise and self.promise()  # Raise previous exception if any.
        result = self.impl(*args, **kwargs)
        self.promise = result
        return result

    def wait(self) -> None:
        return self.impl.wait()

    def close(self) -> None:
        self.impl.close()


class Parallel:
    def __init__(self, actor: Racecar, strategy: str):
        self.worker = Worker(bind(self._respond, actor), strategy, state=True)
        self.callables: Dict[str, Any] = {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()

            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)

            return self.worker(PMessage.READ, name)()

        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _respond(actor: Racecar, state, message: PMessage, name: str, *args, **kwargs):
        state = state or actor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(getattr(state, name))
        elif message == PMessage.CALL:
            result = getattr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
        return state, result


class Damy:
    def __init__(self, env: Racecar):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()
