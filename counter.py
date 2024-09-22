from ultralytics import YOLO
from object_counter import ObjectCounter
import pyzed.sl as sl
import cv2
import argparse
import logging
import asyncio
import time

# Global variable to track if anyone is in frame
isAnyoneInFrame = True
last_person_seen = time.time()
is_time_reached = False
has_timer_started = False

async def monitor_ifAnyoneInFrame():
    print('Started monitoring...')
    global isAnyoneInFrame, last_person_seen, is_time_reached
    while True:
        await asyncio.sleep(1)  # Check every second
        print(f"{time.time()} :: {last_person_seen}")
        if not isAnyoneInFrame and (time.time() - last_person_seen) > 5:
            print("No one in frame for 5 seconds")
            is_time_reached = True

async def main():
    global isAnyoneInFrame, last_person_seen, is_time_reached, has_timer_started
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    # ZED Configuration
    line_points = []
    line_direction = opt.line_direction
    roi = opt.roi
    class_names = {0: 'person'}
    classes_to_count = [0]

    # Initialize ZED
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.sdk_verbose = 1
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 15

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open the ZED camera")
        return

    # Retrieve initial frame size and define ROI
    image = sl.Mat()
    zed.retrieve_image(image, sl.VIEW.LEFT)
    image_data = image.get_data()

    view_height = image_data.shape[0]
    view_width = image_data.shape[1]
    center_x, center_y = view_width // 2, view_height // 2

    if roi == 'polygon' and line_direction == 'vertical':
        line_points = [(center_x - 50, 0), (center_x - 50, view_height), (center_x + 50, view_height), (center_x + 50, 0)]
    elif roi == 'polygon' and line_direction == 'horizontal':
        line_points = [(0, center_y - 50), (0, center_y + 50), (view_width, center_y + 50), (view_width, center_y - 50)]
    elif roi == 'line' and line_direction == 'vertical':
        line_points = [(center_x, 0), (center_x, view_height)]
    elif roi == 'line' and line_direction == 'horizontal':
        line_points = [(0, center_y), (view_width, center_y)]

    # YOLO Configuration
    model = YOLO(opt.weights)
    counter = ObjectCounter(view_img=True,
                            reg_pts=line_points,
                            names=class_names,
                            draw_tracks=False,
                            line_thickness=2,
                            line_dist_thresh=15,
                            output_path=opt.output_path)

    # Start monitoring task
    timer_task = None

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            image_left = sl.Mat()
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            im0 = image_left.get_data()

            if im0.shape[2] == 4:
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGBA2RGB)

            if isAnyoneInFrame:
                last_person_seen = time.time()
                is_time_reached = False
                has_timer_started = False
                if timer_task is not None and not timer_task.done():
                    timer_task.cancel()

            if not isAnyoneInFrame and is_time_reached:
                print("Timer reached, resetting tracking")
                # tracks = model.track(im0, persist=False, show=False, classes=classes_to_count)
                counter.clear_counts()
            # else:
            tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

            isAnyoneInFrame = tracks[0].boxes.id is not None

            if not isAnyoneInFrame and not has_timer_started:
                timer_task = asyncio.create_task(monitor_ifAnyoneInFrame())
                has_timer_started = True

            im0 = counter.start_counting(im0, tracks, line_direction, opt.save_interval, opt.write_batch_interval)

            await asyncio.sleep(0)  # Yield control back to the event loop

    # Close resources
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--line_direction', type=str, default='horizontal', help="'vertical' for vertical, 'horizontal' for horizontal")
    parser.add_argument('--roi', type=str, default='line', help="'polygon' for polygon, 'line' for straight line")
    parser.add_argument('--output_path', type=str, default=None, help="Output path for saved frames")
    parser.add_argument('--save_interval', type=int, default=1, help="Interval (in seconds) to save frames")
    parser.add_argument('--write_batch_interval', type=int, default=5, help="Interval (in seconds) for writing the batch to memory")
    opt = parser.parse_args()

    asyncio.run(main())

