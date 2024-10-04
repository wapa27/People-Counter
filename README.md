# ZED SDK - People Counter
## Setup
```
In a virtual environment, install python 3.10, ZED SDK, and all necessary dependencies. Stay in this environment for running the application.
```
## NOTE: 
```
In order to run correctly, Ultralytics' ObjectCounter will need to updated with the object_counter.py code in this repo - simply put counter.py in the same directory as the new object_counter.py; I updated the import accordingly
```
## Run the program
```
python3 people_counter.py --weights yolov8m.pt --roi [line or polygon] --line_direction [vertical or horizontal] --output_path [path/to/output] --save-interval [number in seconds] --write_batch_interval [number in seconds]
```


