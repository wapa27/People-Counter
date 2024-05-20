from ultralytics import YOLO
from ultralytics.solutions import object_counter
import pyzed.sl as sl
import cv2
import argparse
import logging



def main():
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    model = YOLO(opt.weights)
    line_points = []
    class_names = {0: 'person'}

     # Configure camera parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Adjust as needed
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER
   
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
        center_x = image_data.shape[1]//2
        line_points = [(center_x, 0), (center_x, view_height)]      # CAN ONLY CALCULATE ONCE WE GRAB A FRAME
  
    
    # Initialize object counter with class names
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False,
                 reg_pts=line_points,
                 classes_names=class_names,
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
            tracks = model.track(im0, persist=False, show=False, classes=[0, 2])

            # Start counting objects using the object counter
            im0 = counter.start_counting(im0, tracks)

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
    opt = parser.parse_args()
    main()