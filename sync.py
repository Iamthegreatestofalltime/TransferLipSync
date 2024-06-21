import os
import subprocess
import urllib.request
import face_alignment

def install_dependencies():
    subprocess.run(["pip3", "install", "--upgrade", "pip"])
    subprocess.run(["brew", "update"])
    subprocess.run(["brew", "install", "ffmpeg"])
    subprocess.run(["brew", "install", "cmake"])  # Install CMake for dlib
    subprocess.run(["brew", "install", "boost"])  # Install Boost for dlib

    # Clone video-retalking repository
    if not os.path.isdir("video-retalking"):
        subprocess.run(["git", "clone", "https://github.com/vinthony/video-retalking.git"])
    
    os.chdir("video-retalking")
    
    # Install requirements for video-retalking
    subprocess.run(["pip3", "install", "-r", "requirements.txt"])

    # Install additional dependencies
    subprocess.run(["pip3", "install", "scikit-image"])
    subprocess.run(["pip3", "install", "face-alignment==1.3.4"])
    subprocess.run(["pip3", "install", "basicsr==1.4.2"])  # Install basicsr for GPEN
    subprocess.run(["pip3", "install", "kornia==0.5.1"])  # Install kornia for face3d
    subprocess.run(["pip3", "install", "facexlib==0.2.5"])  # Install facexlib for GFPGAN
    subprocess.run(["pip3", "install", "dlib==19.24.0"])  # Install dlib for GFPGAN
    subprocess.run(["pip3", "install", "face-alignment==1.3.4"])  # Install dlib for GFPGAN

    # Download pre-trained models
    checkpoint_dir = './checkpoints'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model_urls = [
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth',
        'https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat'
    ]
    
    for url in model_urls:
        filename = os.path.join(checkpoint_dir, os.path.basename(url))
        if not os.path.isfile(filename):
            urllib.request.url.retrieve(url, filename)
    
    # Unzip BFM.zip
    subprocess.run(["unzip", "-o", "-d", "./checkpoints/BFM", "./checkpoints/BFM.zip"])

def prepare_video_and_audio(input_video, input_audio):
    video_path = os.path.abspath(input_video)
    audio_path = os.path.abspath(input_audio)
    
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    
    print(f"Video Path: {video_path}")
    print(f"Audio Path: {audio_path}")
    
    return video_path, audio_path

def run_video_retalking(video_path, audio_path):
    temp_output_file_path = 'results/output.mp4'
    final_output_file_path = 'results/output_concat_input.mp4'

    # Run the inference
    cmd = [
        "python3", "inference.py",
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", temp_output_file_path
    ]
    print(f"Running video-retalking with command: {' '.join(cmd)}")
    subprocess.run(cmd)

    if os.path.exists(temp_output_file_path):
        print("Combining input video and generated video...")
        combine_videos(video_path, temp_output_file_path, audio_path, final_output_file_path)
        if os.path.exists(final_output_file_path):
            print("Final Video Preview")
            print(f"Download this video from {final_output_file_path}")
        else:
            print("Combining videos failed.")
    else:
        print("Processing failed. Output video not found.")

def combine_videos(input_video, generated_video, input_audio, output_path):
    import cv2

    def read_video(vid_name):
        video_stream = cv2.VideoCapture(vid_name)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
        return full_frames, fps

    input_video_frames, fps = read_video(input_video)
    output_video_frames, _ = read_video(generated_video)

    frame_h, frame_w = input_video_frames[0].shape[:-1]
    temp_concat_path = './temp/temp/result_concat.mp4'
    if not os.path.isdir('./temp/temp'):
        os.makedirs('./temp/temp')
    out_concat = cv2.VideoWriter(temp_concat_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w*2, frame_h))
    for i in range(len(output_video_frames)):
        frame_input = input_video_frames[i % len(input_video_frames)]
        frame_output = output_video_frames[i]
        out_concat.write(cv2.hconcat([frame_input, frame_output]))
    out_concat.release()

    command = f'ffmpeg -loglevel error -y -i {input_audio} -i {temp_concat_path} -strict -2 -q:v 1 {output_path}'
    subprocess.call(command, shell=True)

def main():
    install_dependencies()

    # Path to input video and audio
    input_video = '../data/DeepSample.mp4'
    input_audio = '../data/audio.wav'

    video_path, audio_path = prepare_video_and_audio(input_video, input_audio)

    run_video_retalking(video_path, audio_path)

if __name__ == "__main__":
    main()