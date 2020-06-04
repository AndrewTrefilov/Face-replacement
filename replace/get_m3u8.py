import cv2
import numpy as np
import ffmpeg
import time
import os
import requests
import sys
import json
import logging
from multiprocessing import Pool


logging.basicConfig(level=logging.INFO)
BATCH_SIZE = 16
COUNT_FRAMES = 125
NUM_PROCESSES = 8


def get_process1(url, headers, num_part):
    start_frame = int(num_part) * COUNT_FRAMES
    end_frame = (int(num_part) + 1) * COUNT_FRAMES
    process1 = (
        ffmpeg.input(filename=url, headers=headers)
        .filter("select", f"between(n,{start_frame}, {end_frame})")
        .filter("setpts", "PTS-STARTPTS")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .run_async(pipe_stdout=True)
    )
    return process1


def green_face(box, frame):
    circle_x = int(box[0] + (box[2] - box[0]) / 2)
    circle_y = int(box[1] + (box[3] - box[1]) / 2)
    frame = cv2.circle(
        frame,
        (circle_x, circle_y),
        max(int(box[2] - box[0]), int(box[3] - box[1])),
        (0, 255, 0),
        -1,
    )
    return frame


def get_process2(width, height, path_to_dir, ts, index):
    if index == 0:
        expr = "N/(25*TB)"
    else:
        expr = f"{index}0/(2*TB)+N/(25*TB)"
    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            hwaccel="cuvid",
        )
        .filter("setpts", expr)
        .output(
            os.path.join(path_to_dir, ts),
            pix_fmt="yuv420p",
            vcodec="h264_nvenc",
            **{"profile:v": "main", "b:v": 1_000_000},
            muxdelay=0,
            r=25,
            loglevel="quiet",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    return process2


def detect_face(detector_id, frame_num, buf):
    if detector_id >= 10:
        detector_url = f"http://detector{detector_id}:90{detector_id}"
    else:
        detector_url = f"http://detector{detector_id}:900{detector_id}"
    if BATCH_SIZE == 1 or (frame_num + BATCH_SIZE) <= COUNT_FRAMES:
        cur_batch_size = BATCH_SIZE
    else:
        cur_batch_size = COUNT_FRAMES - frame_num
    headers = {"BATCH_SIZE": f"{cur_batch_size}"}
    res = requests.post(detector_url, data=buf, headers=headers)

    dets = json.loads(res.content)
    dets = np.asarray(dets)
    tuple_dets = tuple(zip(range(frame_num, frame_num + cur_batch_size), dets))
    return tuple_dets


def generate_job(frames, frame_det=4):
    if BATCH_SIZE == 1:
        frames_for_det = [[frame_num, el] for frame_num, el in enumerate(frames)]
    else:
        frames_for_det = [
            [frame_num, b"".join(frames[frame_num: frame_num + BATCH_SIZE])]
            for frame_num in range(0, COUNT_FRAMES, BATCH_SIZE)
        ]

    if frame_det != 1:
        frames_for_det = frames_for_det[frame_det - 1:: frame_det]

    for idx, el in enumerate(frames_for_det):
        detector_id = idx % NUM_PROCESSES
        el.insert(0, detector_id)

    return frames_for_det


def main():
    width, height = 1280, 720
    path_to_dir, token, index, ts = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )
    date_ts = ts.split("_")[1].split("-")[::-1]
    date_ts = "/".join(date_ts)
    base_url = "https://" + date_ts + "/"
    headers = f"Authorization: Bearer {token}"
    pool = Pool(processes=NUM_PROCESSES)

    url = base_url + ts[1:]

    process1 = get_process1(url, headers, ts[:1])
    frames = []
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frames.append(in_bytes)
    process1.wait()

    frames_for_det = generate_job(frames, frame_det=4)
    jobs = pool.starmap_async(detect_face, frames_for_det)
    jobs = dict([k[i] for k in jobs.get() for i in range(len(k))])
    pool.close()

    # new_dict = jobs.copy()
    # for key in jobs:
    #     dets = jobs[key]
    #     if dets is not None:
    #         for f_num in range(key, key + 18):
    #             if f_num not in jobs:
    #                 new_dict[f_num] = dets.copy()
    #             else:
    #                 dets2 = jobs[f_num]
    #                 if dets2 is not None:
    #                     new_dict[f_num] = np.vstack((dets2, dets))
    #                 else:
    #                     new_dict[f_num] = dets

    process2 = get_process2(width, height, path_to_dir, ts, index)
    for frame_num, frame in enumerate(frames):
        if frame_num in jobs:
            dets = jobs[frame_num]
            if dets is not None:
                frame = np.frombuffer(frame, np.uint8).reshape([height, width, 3])
                for box in dets:
                    frame = green_face(box, frame)
                frame = frame.astype(np.uint8).tobytes()
        process2.stdin.write(frame)
    process2.stdin.close()
    process2.wait()


if __name__ == "__main__":
    t_begin = time.time()
    main()
    t_end = time.time()
    print("ALL TIME:", t_end - t_begin)
