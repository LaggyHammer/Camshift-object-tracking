# Camshift-object-tracking
Object tracking via openCV's Camshift algorithm implementation on a web-cam feed or a video.

This camshift algorithm implementation works via tracking the 'hue' component of the object to be tracked.
As a result, the algorithm performs the best under even lighting on objects that are mostly of a single color shade.

## Usage
```commandline
python object_tracking.py
python object_tracking.py --video test.avi
```
