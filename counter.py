from ultralytics import YOLO
from object_counter import ObjectCounter
import pyzed.sl as sl
import cv2
import argparse
import logging
import time



def main():
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    # _____________ZED CONFIGURATION_______________
    line_points = []
    line_direction = opt.line_direction
    roi = opt.roi
    class_names = {0: 'person'}

    # Init params
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO  # Adjust as needed
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.sdk_verbose = 1
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 30
    # Initialize the ZED camera
    zed = sl.Camera()
    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open the ZED camera")
        exit()
    else:
        image = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        image_data = image.get_data()
        
        view_height = image_data.shape[0]
        view_width = image_data.shape[1]

        center_x = image_data.shape[1]//2
        center_y = image_data.shape[0]//2

        if roi == 'polygon' and line_direction == 'vertical':
            line_points = [(center_x-50, 0), (center_x-50, view_height), (center_x+50, view_height), (center_x+50, 0)]
        elif roi == 'polygon' and line_direction == 'horizontal':
            line_points = [(0, center_y-50), (0, center_y+50),(view_width, center_y+50), (view_width, center_y-50)]
        elif roi == 'line' and line_direction == 'vertical':
            line_points = [(center_x, 0), (center_x, view_height)]
        elif roi == 'line' and line_direction == 'horizontal':
            line_points = [(0, center_y), (view_width, center_y)]
        
    
    # ____________YOLO CONFIGURATION_______________
    model = YOLO(opt.weights)
    classes_to_count = [0]  # person and car classes for count
    # Initialize object counter with class names
    counter = ObjectCounter(view_img=False,
                 reg_pts=line_points,
                 names=class_names,
                 draw_tracks=False,
                 line_thickness=2,
                 line_dist_thresh=15)

    

    # Main loop for processing frames
    while True:
        # Grab a frame from the ZED camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            image_left = sl.Mat()
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            im0 = image_left.get_data()
            
            # Convert image to RGB if it has 4 channels (RGBA)
            if im0.shape[2] == 4:
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGBA2RGB)

            # Perform object tracking with YOLO
            tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

            # Start counting objects using the object counter
            im0 = counter.start_counting(im0, tracks, line_direction)
            # time.sleep(1)  

            # Display the frame
            cv2.imshow("ZED | Object Counting", im0)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--line_direction', type=str, default='horizontal', help="'vertical' for vertical line, 'horizontal' for horizontal line")
    parser.add_argument('--roi', type=str, default='line', help="'polygon' for polygon, 'line' for straight line")
    opt = parser.parse_args()
    main()