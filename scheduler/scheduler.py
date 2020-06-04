import time
import shutil
import logging
import schedule
from redis import Redis
from api.worker import KillWorker
from api import settings

conn = Redis(settings.RQ_DASHBOARD_REDIS_HOST, settings.RQ_DASHBOARD_REDIS_PORT)

logging.basicConfig(level=logging.INFO)


def job():
    while True:
        states = []
        workers = KillWorker.all(connection=conn)
        for worker in workers:
            states.append(worker.state)
        logging.info(states)
        states = list(set(states))
        if len(states) == 1 and states[0] == "idle":
            logging.info("Work done")
            PATH = "../tmpfs_fr"
            shutil.rmtree(PATH, ignore_errors=True)
            break
        time.sleep(1)


if __name__ == "__main__":
    # schedule.every(1).minutes.do(job)
    schedule.every().day.at("02:00").do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)
