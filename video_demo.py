import argparse
import time

import cv2
import torch

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', type=str, help='camera device id')
    parser.add_argument('--config', default='configs/scrfd/scrfd_500m.py', type=str, help='test config file path')
    parser.add_argument('--checkpoint', default='weights/scrfd_500m.pth', type=str, help='checkpoint file')
    parser.add_argument('--device', type=str, default='cuda', help='CPU/CUDA device option')
    parser.add_argument('--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)
    cap = cv2.VideoCapture(args.video)

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap_fps = cap.get(cv2.CAP_PROP_FPS)
    # output_file_path = 'output/result.mp4'
    # video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(output_file_path, video_writer_fourcc, cap_fps, (int(width), int(height)))

    print('Press "Esc", "q" or "Q" to exit.')

    tic = time.time()
    while True:
        ret_val, img = cap.read()
        if not ret_val:
            break

        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        final_image_result = model.show_result(img, result, score_thr=args.score_thr, wait_time=1, show=False)
        video_writer.write(final_image_result)

    toc = time.time()
    print('total time:', toc - tic, 'secs')
    # video_writer.release()
    print('finish!')


if __name__ == '__main__':
    main()
