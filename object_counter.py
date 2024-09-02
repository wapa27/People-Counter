# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        count_reg_color=(255, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            count_reg_color (tuple): RGB color of the counting region.
            count_txt_color (tuple): RGB color of the count text.
            count_bg_color (tuple): RGB color of the count text background.
            line_thickness (int): Line thickness for bounding boxes.
            track_thickness (int): Thickness of the track lines.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
            view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
            track_color (tuple): RGB color of the tracks.
            region_thickness (int): Thickness of the object counting region.
            line_dist_thresh (int): Euclidean distance threshold for line counter.
            cls_txtdisplay_gap (int): Display gap between each class count.
        """
        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
        self.line_dist_thresh = line_dist_thresh
        self.counting_region = None
        self.region_color = count_reg_color
        self.region_thickness = region_thickness

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.names = names  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)
        
        # Custom
        self.p_xyxy = defaultdict(list) #REMOVE????
        self.p_top_right = None
        self.p_top_left = None
        self.p_bottom_left = None
        self.p_bottom_right = None
        self.id_location_mapper = defaultdict()

        # Initialize counting region
        if len(self.reg_pts) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(self.reg_pts)
        elif len(self.reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks, line_direction):
        """Extracts and processes tracks for object counting in a video stream."""
        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()
            confs = tracks[0].boxes.conf.cpu().tolist() # Added to get confidence %

            # Extract tracks
            
            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]} {track_id} {round(conf*100)}%", color=colors(int(track_id), True))

                # Extract and print bounding box coordinates
                x1, y1, x2, y2 = box.tolist()
                top_left = (x1, y1)
                top_right = (x2, y1)
                bottom_left = (x1, y2)
                bottom_right = (x2, y2)
                
                # reg_pts[0]  -------- reg_pts[3]
                #     |                     |
                #     |                     |
                #     |                     |
                #     |                     |
                # reg_pts[1]  -------- reg_pts[2]
                
                roi_min_x = -1
                roi_min_y = -1
                roi_max_x = -1
                roi_max_y = -1
                
                bounding_box_min_x = -1
                bounding_box_min_y = -1
                bounding_box_max_x = -1
                bounding_box_max_y = -1
                
                
                if (len(self.reg_pts) == 2):
                    roi_min_x = self.reg_pts[0][0] if self.reg_pts[0][0] < self.reg_pts[1][0] else self.reg_pts[1][0]
                    roi_max_x = self.reg_pts[0][0] if self.reg_pts[0][0] > self.reg_pts[1][0] else self.reg_pts[1][0]
                    roi_min_y = self.reg_pts[0][1] if self.reg_pts[0][1] < self.reg_pts[1][1] else self.reg_pts[1][1]
                    roi_max_y = self.reg_pts[0][1] if self.reg_pts[0][1] > self.reg_pts[1][1] else self.reg_pts[1][1]
                    
                    bounding_box_min_x = top_left[0] if top_left[0] < bottom_left[0] else bottom_left[0]
                    bounding_box_max_x = top_right[0] if top_right[0] > bottom_right[0] else bottom_right[0] 
                    bounding_box_min_y = top_left[1] if top_left[1] < bottom_left[1] else bottom_left[1]
                    bounding_box_max_y = top_right[1] if top_right[1] > bottom_right[1] else bottom_right[1]
                elif (len(self.reg_pts) >= 3):
                    roi_min_x = self.reg_pts[0][0] if self.reg_pts[0][0] < self.reg_pts[1][0] else self.reg_pts[1][0]
                    roi_max_x = self.reg_pts[2][0] if self.reg_pts[2][0] > self.reg_pts[3][0] else self.reg_pts[3][0]
                    roi_min_y = self.reg_pts[0][1] if self.reg_pts[0][1] < self.reg_pts[3][1] else self.reg_pts[3][1]
                    roi_max_y = self.reg_pts[1][1] if self.reg_pts[1][1] > self.reg_pts[2][1] else self.reg_pts[2][1]
                    
                    bounding_box_min_x = top_left[0] if top_left[0] < bottom_left[0] else bottom_left[0]
                    bounding_box_max_x = top_right[0] if top_right[0] > bottom_right[0] else bottom_right[0]
                    bounding_box_min_y = top_left[1] if top_left[1] < bottom_left[1] else bottom_left[1]
                    bounding_box_max_y = top_right[1] if top_right[1] > bottom_right[1] else bottom_right[1]
                
                # Previous bounding box coordinates
                if self.p_xyxy.get(track_id) is not None:
                    p_x1, p_y1, p_x2, p_y2 = self.p_xyxy[track_id][-1].tolist()
                    # p_top_left = (p_x1, p_y1)
                    # p_top_right = (p_x2, p_y1)
                    # p_bottom_left = (p_x1, p_y2)
                    # p_bottom_right = (p_x2, p_y2)
                    
                if(len(self.reg_pts) == 2 and track_id not in self.id_location_mapper and line_direction == 'vertical'):
                    if (bounding_box_min_x > roi_max_x):
                        self.id_location_mapper[track_id] = "right"
                    elif (bounding_box_max_x < roi_min_x):
                        self.id_location_mapper[track_id] = "left"
                elif(len(self.reg_pts) == 2 and track_id not in self.id_location_mapper and line_direction == 'horizontal'):
                    if (bounding_box_min_y > roi_max_y):
                        self.id_location_mapper[track_id] = "bottom"
                    elif (bounding_box_max_y < roi_min_y):
                        self.id_location_mapper[track_id] = "top"
                elif(len(self.reg_pts) == 4 and track_id not in self.id_location_mapper and line_direction == 'vertical'):
                    if (bounding_box_min_x > roi_max_x):
                        self.id_location_mapper[track_id] = "right"
                    elif (bounding_box_max_x < roi_min_x):
                        self.id_location_mapper[track_id] = "left"
                elif(len(self.reg_pts) == 4 and track_id not in self.id_location_mapper and line_direction == 'horizontal'):
                    if (bounding_box_min_y > roi_max_y):
                        self.id_location_mapper[track_id] = "bottom"
                    elif (bounding_box_max_y <roi_min_y):
                        self.id_location_mapper[track_id] = "top"
                        
                        
                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                # # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color or colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )
                    
                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                if prev_position is None: continue
                
                # # Count objects in any polygon
                # if len(self.reg_pts) >= 3 and line_direction == 'vertical':
                #         if ((top_left[0] > self.reg_pts[3][0] or bottom_left[0] > self.reg_pts[3][0])
                #             and self.id_location_mapper[track_id] != "right"):
                #             self.in_counts += 1
                #             self.class_wise_count[self.names[cls]]["IN"] += 1
                #             self.id_location_mapper[track_id] = "right"
                            
                #         elif ((top_right[0] < self.reg_pts[0][0] or bottom_right[0] < self.reg_pts[0][0])
                #             and self.id_location_mapper[track_id] != "left"):
                #             self.out_counts += 1
                #             self.class_wise_count[self.names[cls]]["OUT"] += 1
                #             self.id_location_mapper[track_id] = "left"
                        
                # elif len(self.reg_pts) >= 3 and line_direction == 'horizontal':
                #         if ((top_left[1] > self.reg_pts[1][1] or top_right[1] > self.reg_pts[1][1])
                #             and self.id_location_mapper[track_id] != "bottom"):
                #             self.in_counts += 1
                #             self.class_wise_count[self.names[cls]]["IN"] += 1
                #             self.id_location_mapper[track_id] = "bottom"
                            
                #         elif ((bottom_left[1] < self.reg_pts[0][1] or bottom_right[1] < self.reg_pts[0][1])
                #             and self.id_location_mapper[track_id] != "top"):
                #             self.out_counts += 1
                #             self.class_wise_count[self.names[cls]]["OUT"] += 1
                #             self.id_location_mapper[track_id] = "top"    
                # # Count objects crossing a line
                # elif len(self.reg_pts) == 2 and line_direction == 'vertical':
                #     # if prev_position is not None and len(self.p_xyxy) > 0:
                #         if (top_left[0] > self.reg_pts[0][0] 
                #             and self.id_location_mapper[track_id] != "right"):
                #             self.in_counts += 1
                #             self.class_wise_count[self.names[cls]]["IN"] += 1
                #             self.id_location_mapper[track_id] = "right"
                #         elif (top_right[0] < self.reg_pts[0][0] 
                #               and self.id_location_mapper[track_id] != "left"):
                #             self.out_counts += 1
                #             self.class_wise_count[self.names[cls]]["OUT"] += 1
                #             self.id_location_mapper[track_id] = "left"
                            
                # elif len(self.reg_pts) == 2 and line_direction == 'horizontal':
                #     # if prev_position is not None and len(self.p_xyxy) > 0:
                #         if ((bottom_left[1] < self.reg_pts[0][1]
                #              or bottom_right[1] < self.reg_pts[0][1]) 
                #             and self.id_location_mapper[track_id] != "top"):
                #             self.in_counts += 1
                #             self.class_wise_count[self.names[cls]]["IN"] += 1
                #             self.id_location_mapper[track_id] = "top"
                #         elif ((top_right[1] > self.reg_pts[0][1] 
                #                 or top_left[1] > self.reg_pts[0][1])
                #               and self.id_location_mapper[track_id] != "bottom"):
                #             self.out_counts += 1
                #             self.class_wise_count[self.names[cls]]["OUT"] += 1
                #             self.id_location_mapper[track_id] = "bottom"
                
                
                # Count objects in any polygon
                if len(self.reg_pts) >= 3 and line_direction == 'vertical':
                        if (bounding_box_min_x > roi_max_x
                            and self.id_location_mapper[track_id] != "right"):
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                            self.id_location_mapper[track_id] = "right"
                            
                        elif (bounding_box_max_x < roi_min_x
                            and self.id_location_mapper[track_id] != "left"):
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1
                            self.id_location_mapper[track_id] = "left"
                        
                elif len(self.reg_pts) >= 3 and line_direction == 'horizontal':
                        if (bounding_box_min_y > roi_max_y
                            and self.id_location_mapper[track_id] != "bottom"):
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                            self.id_location_mapper[track_id] = "bottom"
                            
                        elif (bounding_box_max_y < roi_min_y
                            and self.id_location_mapper[track_id] != "top"):
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1
                            self.id_location_mapper[track_id] = "top"    
                # Count objects crossing a line
                elif len(self.reg_pts) == 2 and line_direction == 'vertical':
                    # if prev_position is not None and len(self.p_xyxy) > 0:
                        if (bounding_box_min_x > roi_max_x 
                            and self.id_location_mapper[track_id] != "right"):
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                            self.id_location_mapper[track_id] = "right"
                        elif (bounding_box_max_x < roi_min_x 
                              and self.id_location_mapper[track_id] != "left"):
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1
                            self.id_location_mapper[track_id] = "left"
                            
                elif len(self.reg_pts) == 2 and line_direction == 'horizontal':
                    # if prev_position is not None and len(self.p_xyxy) > 0:
                        if (bounding_box_max_y < roi_min_y 
                            and self.id_location_mapper[track_id] != "top"):
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                            self.id_location_mapper[track_id] = "top"
                        elif (bounding_box_min_y > roi_max_y
                              and self.id_location_mapper[track_id] != "bottom"):
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1
                            self.id_location_mapper[track_id] = "bottom"
                
                if(self.p_xyxy.get(track_id) is None):
                    self.p_xyxy[track_id] = [box]
                else:
                    self.p_xyxy[track_id].append(box)
                    if(len(self.p_xyxy[track_id])>30):
                        self.p_xyxy[track_id].pop(0)

        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

        if labels_dict:
            self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks, line_direction):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks, line_direction)  # draw region even if no objects

        if self.view_img:
            self.display_frames()
        return self.im0

if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(classes_names)