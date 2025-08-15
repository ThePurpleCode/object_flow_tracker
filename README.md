Optical Flow Tracker:       
Python-based motion tracking in computer vision using the Lucas–Kanade optical flow method with Shi–Tomasi corner detection.
The script tracks feature points across video frames and visualizes motion trajectories, useful for motion analysis, object tracking baselines, and video diagnostics.

Features:      
Lucas–Kanade pyramidal optical flow (cv2.calcOpticalFlowPyrLK)
Shi–Tomasi corner detection for robust feature selection (cv2.goodFeaturesToTrack)
Visual motion trajectories with persistent line overlays and point markers
Auto-reinitialization of features if points are lost
Lightweight, easy to extend for object / motion analytics

Requirements:    
Python 3.7+  
OpenCV (opencv-python)  
NumPy  
