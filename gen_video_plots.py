import os, cv2, json, argparse, glob
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


def gen_graph_video(scores_csv: str, path_video: str, metric: str) -> None:
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

    if metric == "focus":
        focus = True

    if metric == "cognitive_demand":
        cognitive_demand = True

    if metric == "both":
        focus = True
        cognitive_demand = True

    # SET UP VIDEO WRITER
    frame_width, frame_height = 1280, 480 #1024, 384
    video_dir = os.path.dirname(path_video)

    # SET UP FIGURE
    grey_color = '#808080'
    dpi = 150
    fig, ax = plt.subplots(figsize=(frame_width/dpi, frame_height/dpi), dpi=dpi)
    plt.tight_layout(pad=1.0)

    # CHANGE FRAME COLOUR
    for spine in ax.spines.values():
        spine.set_edgecolor(grey_color)

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # DEFINE VIDEO PATHS
    VIDEO_GRAPH_PATH = os.path.join(video_dir, "video_graph.mp4")
    VIDEO_MONTAGE_PATH = os.path.join(video_dir, "video_montage.mp4")
    VIDEO_PADDED_PATH = os.path.join(video_dir, "video_padded.mp4")
    #out = cv2.VideoWriter(VIDEO_GRAPH_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))

    # PLOT THE SCORES
    for t in range(len(x_data)):
        ax.clear()

        if focus:
            ax.plot(x_data, y_data[:, 0], linestyle='-', lw=1.5, color='#C5ABFD', alpha=1., label="Cognitive Demand")
        if cognitive_demand:
            ax.plot(x_data, y_data[:, 1], linestyle='-', lw=1.5, color='#FFD04B', alpha=1., label="Focus")

        # ADJUSTMENT TO AVOID RESIZING DUE TO ANNOTATIONS
        y_pos_cog_demand = y_data[t, 0]
        y_pos_focus = y_data[t, 1]
        threshold = 95
        if y_pos_cog_demand > threshold:
            y_pos_cog_demand -= 10
        if y_pos_focus > threshold:
            y_pos_focus -= 10 

        # ADD DYNAMIC MARKERS
        if cognitive_demand:
            ax.scatter(x_data[t], y_data[t, 0], color='#C5ABFD', marker="o", s=20)
            offset_x = 0.005 * (max(x_data) - min(x_data))
            label_x_position = x_data[t] + offset_x if t <= num_frames_video / 2 else x_data[t] - offset_x
            offset_y = 0.005 * (max(y_data[:,0]) - min(y_data[:,0]))
            label_y_position = y_pos_cog_demand + offset_y
            
            ax.text(label_x_position, label_y_position, f"Cognitive Demand: {y_data[t, 0]:.0f}", fontsize=8, verticalalignment='bottom', 
                    ha='right' if t > num_frames_video / 2 else 'left', color=grey_color, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

        if focus:
            ax.scatter(x_data[t], y_data[t, 1], color='#FFD04B', marker="D", s=20)
            offset_x = 0.005 * (max(x_data) - min(x_data))
            label_x_position = x_data[t] + offset_x if t <= num_frames_video / 2 else x_data[t] - offset_x            
            offset_y = 0.005 * (max(y_data[:,1]) - min(y_data[:,1]))
            label_y_position = y_pos_focus + offset_y
            
            ax.text(label_x_position, label_y_position, f"Focus: {y_data[t, 1]:.0f}", 
                    fontsize=8, verticalalignment='bottom', 
                    ha='right' if t > num_frames_video / 2 else 'left', color=grey_color, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

        # SET UP AXES
        ax.set_ylim(0., 105.)
        ax.set_xlim(0, num_frames_video/FPS)
        ax.set_xticks(np.arange(0, num_frames_video/FPS, 2))
        ax.set_xticklabels([f"{int(t//60):02d}:{int(t%60):02d}" for t in np.arange(0, num_frames_video/FPS, 2)])
        ax.tick_params(axis='x', colors="darkgrey")
        ax.tick_params(axis='y', colors="darkgrey")
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        # ADD HORIZONTAL GRID ONLY
        ax.yaxis.grid(color='lightgrey', linestyle='-', alpha=0.35)

        # INSERT LEGEND
        plt.subplots_adjust(bottom=0.0001)
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2, framealpha=0.7,
                           facecolor='white', frameon=True)
        legend.get_frame().set_linewidth(0)
        plt.tight_layout(rect=[0, 0.0001, 1, 1])
        '''
        # INSERT LEGEND
        legend = ax.legend(loc='upper center', fontsize=8, ncol=2)
        legend.get_frame().set_alpha(0.)
        #legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(0)
        '''
        for text in legend.get_texts():
            text.set_color(grey_color)

        # SAVE CURRENT FIGURE AS AN IMAGE
        temp_img_path = os.path.join(video_dir, f'temp_plot_{t:04d}.png')
        plt.savefig(temp_img_path, dpi=dpi) #bbox_inches='tight'

    # RELEASE VIDEO WRITER AND CLOSE FIGURE
    plt.close()

    # CALCULATE PADDING DIMENSIONS
    pad_width = max(frame_width, width)
    pad_height = max(frame_height, height)

    # GENERATE VIDEO GRAPH
    sp.call([
        'ffmpeg',
        '-r', str(FPS),
        '-f', 'image2',
        '-s', f'{frame_width}x{frame_height}',
        '-y',
        '-i', os.path.join(video_dir, 'temp_plot_%04d.png'),
        '-vcodec', 'libx264',
        '-crf', '18',
        '-pix_fmt', 'yuv420p', VIDEO_GRAPH_PATH,
        '-loglevel', 'panic'
    ])

    # PAD THE VIDEO
    sp.call([
        'ffmpeg',
        '-y',
        '-i', path_video,
        '-vf', 'pad=width={}:height={}:x=(ow-iw)/2:y=(oh-ih)/2:color=white'.format(pad_width, pad_height), VIDEO_PADDED_PATH,
        '-loglevel', 'panic', 
    ])

    # STACK THE PADDED VIDEO AND GRAPH
    sp.call([
        'ffmpeg',
        '-y',
        '-i', VIDEO_PADDED_PATH,
        '-i', VIDEO_GRAPH_PATH,
        '-filter_complex', 'vstack', VIDEO_MONTAGE_PATH,
        '-loglevel', 'panic'
    ])

    # DELETE TEMP FILES
    frame_files = glob.glob(os.path.join(video_dir, 'temp_plot_*.png'))
    for file in frame_files:
        os.remove(file)

    os.remove(VIDEO_PADDED_PATH)
    os.remove(VIDEO_GRAPH_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_video_heatmap', type=str, help='Path to video heatmap')
    parser.add_argument('--path_scores_csv', type=str, help='Path to CSV file containing frame-level scores')
    parser.add_argument('--metric', type=str, help='Cognitive Demand or Focus or both', default='both')

    opt = parser.parse_args()
    gen_graph_video(opt.path_scores_csv, opt.path_video_heatmap, opt.metric)