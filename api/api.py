import os
import re
import logging
import settings
import subprocess
import psutil
import requests
import rq_dashboard
from uuid import uuid4
from worker import conn, KillQueue
from flask import Flask, request, send_from_directory

logging.basicConfig(level=logging.INFO)
os.system("/sbin/ldconfig")
app = Flask(__name__)
app.config.from_object(rq_dashboard.default_settings)
app.config.from_object(settings)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")

q = KillQueue(name="default", connection=conn)


PATH = "../tmpfs_fr"


def get_archive_playlist(camera_id, start_time, stop_time, token):
    url = f"https://{camera_id}.m3u8?START_TIME={start_time}&STOP_TIME={stop_time}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    return response


def duplicate_segment(playlist):
    orig_pl = playlist.splitlines()
    segments_list = []
    header_list = []
    for el in orig_pl:
        if "#EXTINF:" in el or "//" in el:
            segments_list.append(el)
        else:
            header_list.append(el)
    changed_sl = []
    for i in range(len(segments_list) - 1):
        if (i + 1) % 2 == 0:
            sn = segments_list[i].split("/")[-1]
            for num_part in range(2):
                changed_sl.append(segments_list[i - 1])
                changed_sl.append(re.sub(sn, str(num_part) + sn, segments_list[i]))
    return "\n".join(header_list + changed_sl)


def replace_playlist(playlist, token):
    parts = re.findall(r"/.+\.ts", playlist)
    for index, part in enumerate(parts):
        playlist = re.sub(
            part, f"/download/{token}/{index}/{part.split('/')[-1]}", playlist
        )
    playlist = re.sub("#EXT-X-TARGETDURATION:10", "#EXT-X-TARGETDURATION:5", playlist)
    playlist = re.sub("#EXTINF:10,", "#EXTINF:5", playlist)
    return playlist


@app.route("/get_playlist", methods=["GET"])
def get_playlist():
    json = request.args.to_dict()
    token = json["token"]

    response = get_archive_playlist(**json)
    if response.status_code == 200:
        playlist = response.text
        playlist = duplicate_segment(playlist)
        playlist = replace_playlist(playlist, token)
        return playlist
    else:
        return "Bad Request", 400


def send_file(path_to_dir, ts):
    return send_from_directory(directory=path_to_dir, filename=ts, as_attachment=True)


def execute_cmd(command):
    process = subprocess.Popen([command], shell=True)
    while process.poll() is None:
        pass
    return


def do_job(path_to_dir, token, index, ts):
    job_id = str(uuid4())
    with open(os.path.join(path_to_dir, "job_ids.txt"), "a+") as f:
        f.write(f"{job_id} {token} {index} {ts}" + "\n")
    cur_job = q.enqueue(
        f=execute_cmd,
        args=[
            f"cd ../replace && python3 get_m3u8.py {path_to_dir} {token} {index} {ts}"
        ],
        result_ttl=5000,
        job_id=job_id,
        ttl=100,
        job_timeout=10,
    )
    while True:
        if cur_job.is_finished:
            return send_file(path_to_dir, ts)
            # return "OK", 200
        elif cur_job.is_failed:
            return "Not Found", 404


@app.route("/download/<token>/<index>/<ts>", methods=["GET"])
def download_segment(token, index, ts):
    path_to_dir = os.path.join(PATH, token)
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir, exist_ok=True)

    path_to_ts = os.path.join(path_to_dir, ts)
    if os.path.exists(path_to_ts):
        return send_file(path_to_dir, ts)
    elif not os.path.exists(path_to_ts):
        return do_job(path_to_dir, token, index, ts)


@app.route("/cancel/<token>/<index>/<ts>", methods=["GET"])
def cancel_job(token, index, ts):
    path_to_dir = os.path.join(PATH, token)
    with open(os.path.join(path_to_dir, "job_ids.txt"), "r") as f:
        job_ids = f.read().splitlines()

    string = [el for el in job_ids if f"{token} {index} {ts}" in el][0]
    cancel_job_id = string.split()[0]
    cancel_job = q.fetch_job(cancel_job_id)
    if cancel_job is not None:
        logging.info(f"CANCEL_JOB is not None")
        logging.info(cancel_job)
        logging.info(cancel_job.get_status())
        logging.info(f"CANCEL_JOB enqueued_at: {cancel_job.enqueued_at}")
        cancel_job.kill()
        logging.info("JOB CANCEL")
        logging.info(cancel_job)
        logging.info(cancel_job.get_status())
        logging.info(f"CANCEL_JOB ended_at: {cancel_job.ended_at}")
    else:
        logging.info(f"CANCEL_JOB is None")

    return "OK", 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
