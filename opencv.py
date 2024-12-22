import argparse
import os
import platform
import sys

import cv2
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import time
from threading import Thread, Event
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
import torch.backends.cudnn as cudnn
import torch
from pathlib import Path
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
import numpy as np  # [추가]
from uuid import uuid4  # [추가] 고유 ID 생성을 위해 사용

import threading

# 추가: 전역 잠금 객체
resource_lock = threading.Lock()


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


app = Flask(__name__)
CORS(app)

stream_threads = {}
stream_events = {}

def process_stream(source, stop_event):
    
    visualize = False 
    conf_thres=0.25
    iou_thres=0.45
    classes=None
    agnostic_nms=False
    max_det=1000
    name='exp'
    exist_ok=False
    save_txt=False
    save_crop=False
    line_thickness=1
    nosave=True
    hide_labels=False
    hide_conf=False
    half=False
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    jpeg_quality = 90  # JPEG 압축 품질 (1-100, 낮을수록 더 낮은 품질)



    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    project=ROOT
   # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device("cpu")
    model = DetectMultiBackend("best.pt", device=device, dnn=False, data="data/water.yaml", fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((320, 320), s=stride)  # check image size
    
    
    view_img = check_imshow()
    cudnn.benchmark = True
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        
        if stop_event.is_set():  # **[추가] 중단 요청 확인**
            dataset.stop()
            break
        
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=False, visualize=False)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            print("_TEST_......")
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #
            im0 = cv2.resize(im0, (320, 320))

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #
            if len(det):
                # 기존
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                # mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                # im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                # annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ----------------------------------------------------------------------------------------
               

                #------------------------
                # 윤곽선 그리기
                #------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]


                overlay = im0.copy()  # 원본 이미지 복사본 생성
                for mask, color in zip(masks, mcolors):

                    #  water_area
                    
                    binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)  # Tensor를 NumPy로 변환
                    resized_mask = cv2.resize(binary_mask, (im0.shape[1], im0.shape[0]))  # 마스크 크기를 원본 이미지 크기로 조정

                    # 윤곽선 찾기
                    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 색상을 OpenCV 형식으로 변환 (BGR)
                    bgr_color = color  # RGB to BGR

                    # 윤곽선 그리기
                    for contour in contours:
                        if cv2.contourArea(contour) > 10:  # 최소 면적 필터링
                            cv2.drawContours(im0, [contour], -1, bgr_color, thickness=1)  
                            # cv2.drawContours(im0, [contour], -1, bgr_color, thickness=2)  

                # # 윤곽선 복사본을 원본에 합성 (깜빡임 제거)
                alpha = 0.0  # 윤곽선 투명도(0.0-1.0)
                cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0, im0)  # 원본과 윤곽선 합성
                #------------------------
                
                
                #------------------------
                # 라벨링 하기
                #------------------------
                # # #Write results
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(f'{txt_path}.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        #라벨표시 여부
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                       
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            
            im0 = annotator.result()
            if view_img:
                
                # Replace cv2.imshow with web streaming preparation
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                _, buffer = cv2.imencode('.jpg', im0)
                frame = buffer.tobytes()
                if buffer is None or len(buffer) == 0:
                    print("Error: Failed to encode frame as JPEG.")
                else:
                    print(f"Frame encoded successfully: {len(buffer)} bytes")

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
    #    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def cleanup_stream_resources(source, stream_id):
    """Clean up resources associated with a stream."""
    with resource_lock:
        if source in stream_threads and stream_id in stream_threads[source]:
            del stream_threads[source][stream_id]
        if stream_id in stream_events:
            del stream_events[stream_id]

    # OpenCV 리소스 해제
    cv2.destroyAllWindows()
    print(f"Resources for stream {stream_id} have been cleaned up.")  # 수정: source_id -> stream_id



def process_stream2(source, stop_event):
    """Non-YOLO stream processing (simple streaming without YOLO detection)."""
    # 동일한 스트리밍 로직을 유지하면서 YOLO 감지 없이 진행
    video_capture = cv2.VideoCapture(source)
    
    # FPS 설정
    fps = 30  # 기본 FPS 30으로 설정
    delay = 1 / fps  # FPS에 따른 프레임 간의 시간 차이 계산

    while not stop_event.is_set():
        ret, frame = video_capture.read()
        if not ret:
            break

        # 여기서는 YOLO 없이 그냥 스트리밍만 합니다.
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # 프레임 간의 시간 차이를 맞추기 위해 지연
        time.sleep(delay)  # FPS에 맞춰 지연 시간 추가

    video_capture.release()
    cv2.destroyAllWindows()



@app.route('/stream-id', methods=['GET'])
def generate_stream_id():
    """Generate a unique stream ID for a given HLS source."""
    source = request.args.get('rtspAddr')
    if not source:
        return jsonify({'error': 'Missing hlsAddr parameter'}), 400

    source_id = str(uuid4())  # Generate a unique ID for the stream
    stop_event = Event()

    with resource_lock:
        if source not in stream_threads:
            stream_threads[source] = {}
        stream_threads[source][source_id] = stop_event
        stream_events[source_id] = stop_event

    return jsonify({'streamId': source_id})



@app.route('/stream/<stream_id>', methods=['GET'])
def stream_video(stream_id):
    """Stream video frames for a given stream ID."""
    source = request.args.get('rtspAddr')
    yolo = request.args.get('yolo')  # yolo 파라미터를 받음 (기본값은 'false')
    print('yolo : ',yolo)
    if not source:
        return jsonify({'error': 'Missing hlsAddr parameter'}), 400

    with resource_lock:
        if source not in stream_threads or stream_id not in stream_threads[source]:
            return jsonify({'error': 'Invalid or expired stream ID'}), 404

        stop_event = stream_threads[source][stream_id]

    def stream_worker():
        """Stream video frames."""
        try:
            if yolo.lower() == 'true':
                # YOLO 모델을 사용하는 스트리밍
                for frame in process_stream(source, stop_event):
                    yield frame
            else:
                # YOLO 모델을 사용하지 않는 스트리밍
                for frame in process_stream2(source, stop_event):
                    yield frame
        except GeneratorExit:
            print(f"Stream worker for {stream_id} exited.")
        finally:
            cleanup_stream_resources(source, stream_id)  # 호출 시 stream_id를 전달

    return Response(stream_worker(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/stop-stream', methods=['POST'])
def stop_stream():
    """Stop a stream using the provided streamId."""
    data = request.json
    stream_id = data.get('streamId')
    if not stream_id:
        return jsonify({'error': 'Missing streamId parameter'}), 400

    with resource_lock:
        if stream_id not in stream_events:
            return jsonify({'error': 'Invalid streamId'}), 404

        stop_event = stream_events[stream_id]
        stop_event.set()  # Trigger the stop event to halt the stream
        del stream_events[stream_id]  # Clean up resources

    return jsonify({'message': f'Stream {stream_id} stopped successfully'})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
