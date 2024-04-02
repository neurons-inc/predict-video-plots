import os, cv2, json, argparse
import numpy as np
import pandas as pd
import subprocess as sp
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", family="serif")

FPS: int = 24


def get_whff(file: str) -> tuple[int, int, int, int]:
    '''
    Get width, height and fps using ffprobe
    '''
    cmd = "ffprobe -v quiet -print_format json -select_streams v:0 -count_packets -show_streams"

    args = cmd.split(' ')
    args.append(file)

    s = sp.check_output(args).decode('utf-8')
    d = json.loads(s)

    for vd in d['streams']:
        if 'width' in vd.keys():
            break

    width = int(vd['width'])
    height = int(vd['height'])
    framerate, timebase = vd['r_frame_rate'].split('/')
    fps = int(round(int(framerate) / int(timebase)))
    frames = int(vd['nb_read_packets'])

    return width, height, fps, frames


def gen_graph_video(scores_csv: str, path_video: str) -> None:
    '''
    Generate the video graph at the desired frame rate of 24 fps,
    ensuring alignment with the relevant video
    '''
    # GET VIDEO INFO
    width, height, fps, num_frames_video = get_whff(path_video)
    assert fps == FPS, "Frame rate mismatch: expected {} fps, got {} fps".format(FPS, fps)
    x_data = np.linspace(0, num_frames_video/FPS, num_frames_video)

    # GET SCORES IN RIGHT FORMAT
    scores = pd.read_csv(scores_csv)
    scores = scores.drop_duplicates(subset=['Frame'], keep='first')
    assert len(scores['Frame']) == num_frames_video, f"Number of frames in scores CSV ({len(scores['Frame'])}) does not match the number of video frames ({num_frames_video})"
    y_data = np.empty([len(scores['Frame']), 2])
    y_data[:,0] = scores["Cognitive Demand"]
    y_data[:,1] = scores["Focus"]

    # SET UP VIDEO WRITER
    frame_width, frame_height, fig_height = 1920, 576, 6 #1024, 576
    video_dir = os.path.dirname(path_video)

    if frame_height == height:
        frame_height, fig_height = int(576/1.5), int(6/1.5)

    # SET UP FIGURE
    fig, ax = plt.subplots(figsize=[20, fig_height])

    # DEFINE VIDEO PATHS
    VIDEO_GRAPH_PATH = os.path.join(video_dir, "video_graph.mp4")
    VIDEO_MONTAGE_PATH = os.path.join(video_dir, "video_montage.mp4")
    VIDEO_PADDED_PATH = os.path.join(video_dir, "video_padded.mp4")
    out = cv2.VideoWriter(VIDEO_GRAPH_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))

    # PLOT THE SCORES
    for t in range(len(x_data)):
        ax.clear()

        # PLOT THE LINES
        ax.plot(x_data[:t], y_data[:t, 0], linestyle='-', lw=2.0, color='#C5ABFD', alpha=1., label="Cognitive Demand")
        ax.plot(x_data[:t], y_data[:t, 1], linestyle='-', lw=2.0, color='#FFD04B', alpha=1., label="Focus")

        # SET UP AXIS
        ax.set_ylim(0., 100.)
        ax.set_xlim(0, num_frames_video/FPS)
        ax.set_xticks(np.arange(0, num_frames_video/FPS, 2))
        ax.set_xticklabels([f"{int(t//60):02d}:{int(t%60):02d}" for t in np.arange(0, num_frames_video/FPS, 2)])

        # ADD GRID
        ax.grid(color='lightgrey', linestyle='-', alpha=0.35)

        # INSERT LEGEND
        legend = ax.legend(loc='upper left', fontsize=12, ncol=2)
        legend.get_frame().set_alpha(0.7)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(0)

        # SAVE CURRENT FIGURE AS AN IMAGE
        temp_img_path = os.path.join(video_dir, 'temp_plot.png')
        plt.savefig(temp_img_path, bbox_inches='tight')

        # READ THE IMAGE AND RESIDE TO DESIRED FRAME SIZE
        frame = cv2.imread(temp_img_path)
        frame = cv2.resize(frame, (frame_width, frame_height))

        # WRITE FRAME TO VIDEO
        out.write(frame)

    # RELEASE VIDEO WRITER AND CLOSE FIGURE
    out.release()
    plt.close()

    # CALCULATE PADDING DIMENSIONS
    pad_width = max(frame_width, width)
    pad_height = max(frame_height, height)

    # PAD THE VIDEO
    sp.call([
        'ffmpeg', '-y', '-i', path_video, '-vf', 'pad=width={}:height={}:x=(ow-iw)/2:y=(oh-ih)/2:color=white'.format(pad_width, pad_height),
        VIDEO_PADDED_PATH, '-loglevel', 'panic', 
    ])

    # STACK THE PADDED VIDEO AND GRAPH
    sp.call([
        'ffmpeg', '-y', '-i', VIDEO_PADDED_PATH, '-i', VIDEO_GRAPH_PATH,
        '-filter_complex', 'vstack', VIDEO_MONTAGE_PATH, '-loglevel', 'panic'
    ])

    # DELETE TEMP FILES
    os.remove(temp_img_path)
    os.remove(VIDEO_PADDED_PATH)
    os.remove(VIDEO_GRAPH_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_video_heatmap', type=str, help='Path to video heatmap')
    parser.add_argument('--path_scores_csv', type=str, help='Path to CSV file containing frame-level scores')

    opt = parser.parse_args()
    gen_graph_video(opt.path_scores_csv, opt.path_video_heatmap)