import os
from api import settings
import sys
import time
import signal
from threading import Thread
from redis import Redis
from rq import Worker, Queue, Connection
from rq.job import Job

listen = ["default"]

os.system("/sbin/ldconfig")
conn = Redis(settings.RQ_DASHBOARD_REDIS_HOST, settings.RQ_DASHBOARD_REDIS_PORT)
kill_key = "rq:jobs:kill"


class KillJob(Job):
    def kill(self):
        """ Force kills the current job causing it to fail """
        if self.is_started:
            self.connection.sadd(kill_key, self.get_id())

    def _execute(self):
        def check_kill(conn, id, interval=1):
            while True:
                res = conn.srem(kill_key, id)
                if res > 0:
                    os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(interval)

        t = Thread(target=check_kill, args=(self.connection, self.get_id()))
        t.start()

        return super()._execute()


class KillQueue(Queue):
    job_class = KillJob


class KillWorker(Worker):
    queue_class = KillQueue
    job_class = KillJob


if __name__ == "__main__":
    worker_name = sys.argv[1]
    with Connection(conn):
        worker = KillWorker(settings.QUEUES, name=worker_name)
        worker.work()
