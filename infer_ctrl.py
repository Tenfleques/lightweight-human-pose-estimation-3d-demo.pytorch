from argparse import ArgumentParser
import json
import os
import traceback

import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def get_best_resize_fit(shape, org_shape, scale=.95):
    new_width = shape[1] * scale
    if new_width > org_shape[1]:
        new_width = org_shape[1]
    
    if scale == 1:
        scale = .9
    
    new_height = int(new_width * shape[0] / shape[1])
    
    shape = int(new_height), int(new_width)
    
    if shape[0] > org_shape[0]:
        return get_best_resize_fit(shape, org_shape, scale=scale)
    
    return shape

def pad_resize_image(im: np.ndarray, target_shape) -> np.ndarray:
    """
        resize frame to some original shape for purporses of writing video frame to an already instantiated video output format
    """
    container = np.full((target_shape[0], target_shape[1], target_shape[2]), 0, dtype=np.uint8)

    if im.shape[0] == target_shape[0] and im.shape[1] == target_shape[1]:
        return im 

    new_height, new_width = get_best_resize_fit(im.shape, target_shape, 1)
    
    im = cv2.resize(im, (new_width, new_height), None)

    container[:im.shape[0], :im.shape[1], :] = im

    return container[:, :, :3]

class InferCtrl:
    net=None
    extrinsics=None

    def __init__(self, model, height=256, device='CPU', openvino=False, tensorrt=False, extrinsics_path="./data/extrinsics.json", fx=1, canvas_shape=(720, 1280, 3)) -> None:
        
        if openvino:
            from modules.inference_engine_openvino import InferenceEngineOpenVINO
            self.net = InferenceEngineOpenVINO(model, device)
        else:
            from modules.inference_engine_pytorch import InferenceEnginePyTorch
            self.net = InferenceEnginePyTorch(model, device, tensorrt)

        try:
            with open(extrinsics_path, 'r') as f:
                self.extrinsics = json.load(f)
        except Exception:
            with open("./data/extrinsics.json", 'r') as f:
                self.extrinsics = json.load(f)
            traceback.print_exc()
        
        self.base_height = height
        self.fx = fx
        self.canvas_3d = np.zeros(canvas_shape, dtype=np.uint8)
        self.plotter = Plotter3d(self.canvas_3d.shape[:2])

        # print("[INFO] plotter shape {}".format(self.plotter.shape))
        print("[INFO] canvas shape {}".format(self.canvas_3d.shape))

    def process_frame(self, frame, inference_result, merged=False):
        poses_3d = inference_result.get("pose_3d", {}).get("value", [])
        poses_2d = inference_result.get("pose_2d", {}).get("value", [])
        edges = inference_result.get("edges", {}).get("value", [])
        
        
        self.plotter.plot(self.canvas_3d, poses_3d, edges)    
        draw_poses(frame, poses_2d)

        if merged:
            frame_side = np.copy(self.canvas_3d)

            new_w = min(frame.shape[1], frame_side.shape[1])

            rel_h_f = int(new_w * frame.shape[0] * 1.0/frame.shape[1])
            rel_h_s = int(new_w * frame_side.shape[0] * 1.0/frame_side.shape[1])

            frame = cv2.resize(frame, (new_w, rel_h_f))
            frame_side = cv2.resize(frame_side, (new_w, rel_h_s))

            return np.hstack([frame, frame_side])

        return frame, np.copy(self.canvas_3d)

    def infer(self, frame, is_video=True, fx=None):
        stride = 8
        output = {}
        if fx is None:
            fx = self.fx
            output["focal_length"] = {
                "value": fx,
                "comment": "default value used because none was supplied"
            }
            
        
        R = np.array(self.extrinsics['R'], dtype=np.float32)
        t = np.array(self.extrinsics['t'], dtype=np.float32)

        input_scale = self.base_height / frame.shape[0]
            
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  

        # scaled_img = pad_resize_image(scaled_img, (scaled_img.shape[0], (scaled_img.shape[1] + stride)//stride, scaled_img.shape[2]))

        output["input_size"] = {
                "value": scaled_img.shape,
                "comment": "network inpute size"
            }

        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

            output["focal_length"] = {
                "value": fx,
                "comment": "Focal length is unknown, 0.8 * frame width used"
            }

        # the inference 
        inference_result = self.net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        edges = []
        
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))

        output["pose_3d"] = {
                "value": poses_3d,
                "comment": "re-oriented 3D poses"
            }

        output["pose_2d"] = {
                "value": poses_2d,
                "comment": "2D poses"
            }
        
        output["edges"] = {
                "value": edges,
                "comment": "2D poses"
            }
        
        return output
        

def main():
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-o', '--output', help='output directory for estimated results', default='./output')

    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--use-tensorrt', help='Optional. Run network with TensorRT as inference engine.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')


    infer_ctrl = InferCtrl(args.model, args.height_size, device=args.device, openvino=args.use_openvino, tensorrt=args.use_tensorrt, extrinsics_path=args.extrinsics_path, fx=args.fx)
    
    frame_provider = ImageReader(args.images)
    is_video = False


    outname = args.images

    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
        try:
            cam_index = int(args.video)
            outname = "output-cam-{}".format(cam_index)
        except:
            outname = ".".join(args.video.split(os.sep)[-1].split(".")[:-1])
        
    fx = args.fx

    mean_time = 0
    i = 0
    frame_size = None, None

    dir_name = os.path.join(args.output, outname)
    os.makedirs(dir_name, exist_ok=True)

    outname_3d = os.path.join(dir_name, "{}-temp-3D.mp4".format(outname))
    outname_frames = os.path.join(dir_name, "{}-temp.mp4".format(outname))
    outname_combined_frames = os.path.join(dir_name, "{}-combined-temp.mp4".format(outname))

    outname_3d_compressed = os.path.join(dir_name, "{}-3D.mp4".format(outname))
    outname_frames_compressed = os.path.join(dir_name, "{}.mp4".format(outname))
    outname_frames_combined_compressed = os.path.join(dir_name, "{}-combined.mp4".format(outname))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    out_3d = None
    out_combined = None
    fps = None
    fps_out = 15
    
    try:
        results_json = []
        for frame in frame_provider:
            current_time = cv2.getTickCount()
            if frame is None:
                break

                
            inference_result = infer_ctrl.infer(frame, is_video=False, fx=fx)          

            # print(inference_result)
            # continue
            # intepreting results
            frame, canvas_3d = infer_ctrl.process_frame(frame, inference_result)
            combined_frame = infer_ctrl.process_frame(frame, inference_result, merged=True)

            for key in inference_result.keys():
                value = inference_result[key].get("value", [])
                
                if "numpy" in str(type(value)):
                    inference_result[key]["value"] = getattr(value, "tolist", lambda: value)()
                
            
            results_json.append(inference_result)

            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()

            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            
            fps = int(1 / mean_time * 10) / 10
            
            cv2.putText(frame, 'processing FPS: {}'.format(fps),
                        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            i += 1
            
            
            if out is None or out_3d is None:
                frame_size = tuple(int(i) for i in frame.shape[:2])
                frame_size = frame_size[::-1]
                
                frame_size_3d = tuple(int(i) for i in canvas_3d.shape)
                frame_size_3d = canvas_3d.shape[1::-1]
                
                frame_size_combined = tuple(int(i) for i in combined_frame.shape)
                frame_size_combined = combined_frame.shape[1::-1]

                print(frame_size_3d, frame_size, frame_size_combined)

                out_3d = cv2.VideoWriter(outname_3d, fourcc, fps_out, frame_size_3d, True)
                out = cv2.VideoWriter(outname_frames,fourcc, fps_out, frame_size, True)
                out_combined = cv2.VideoWriter(outname_combined_frames, fourcc, fps_out, frame_size_combined, True)
            
            if out is not None:
                out.write(frame)

            if out_3d is not None:
                out_3d.write(canvas_3d)

            if out_combined is not None:
                out_combined.write(combined_frame)
     
    except KeyboardInterrupt:
        print("[INFO] interrupted")
        
    if out is not None:
        out.release()

    if out_3d is not None:
        out_3d.release()

    with open(os.path.join(dir_name, "results.json"), "w") as fp:
        json.dump(results_json, fp)
        fp.close()

    try:
        os.system(f"ffmpeg -i {outname_frames} -loglevel error -vcodec libx264 {outname_frames_compressed}")
        os.system(f"ffmpeg -i {outname_3d} -loglevel error -vcodec libx264 {outname_3d_compressed}")
        os.system(f"ffmpeg -i {outname_combined_frames} -loglevel error -vcodec libx264 {outname_frames_combined_compressed}")

        # os.system(f"rm -rf {outname_frames} {outname_3d}")
    except:
        traceback.print_exc()

    print("[INFO] finished .... ")


if __name__ == '__main__':
    main()
