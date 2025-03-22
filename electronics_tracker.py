import cv2
import numpy as np
from datetime import datetime
import threading
import time

class ElectronicsTracker:
    def __init__(self):
        self.camera = None
        self.rf_sensor_active = False
        self.detected_devices = set()
        
    def initialize_camera(self):
        """Initialize the camera for visual detection"""
        try:
            # Try different camera indices with DirectShow backend
            for camera_index in range(2):
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if self.camera.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = self.camera.read()
                    if ret:
                        print(f"Successfully initialized camera {camera_index}")
                        return
                    self.camera.release()
            raise Exception("No working camera found")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            raise
        
    def initialize_rf_sensor(self):
        """Initialize RF sensor for smartphone detection
        Note: This is a placeholder for actual RF sensor implementation
        """
        # TODO: Implement actual RF sensor initialization
        self.rf_sensor_active = True
        
    def detect_devices_vision(self, frame):
        """Detect electronic devices using computer vision"""
        # Convert frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhanced image processing for device detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Increased blur kernel for noise reduction
        edges = cv2.Canny(blurred, 10, 50)  # Even lower thresholds for more edge detection
        
        # Enhance edges using morphological operations
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_devices = []
        
        # Process each contour
        for contour in contours:
            # Filter contours based on area
            area = cv2.contourArea(contour)
            if area > 300:  # Further lowered minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h
                
                # Calculate shape complexity using contour approximation
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Calculate solidity (area ratio)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Calculate rectangularity
                rect_area = w * h
                rectangularity = float(area) / rect_area if rect_area > 0 else 0
                
                # Enhanced visibility analysis for device detection
                roi = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Calculate color variance to detect uniform light sources
                color_variance = np.std(roi_color)
                
                # Calculate edge density with more weight on strong edges
                edge_roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(edge_roi > 0) / (w * h)
                
                # Calculate texture complexity
                texture_score = np.std(roi) * edge_density
                
                # Calculate visibility score with reduced sensitivity to light
                visibility_score = (texture_score * color_variance) / (255.0 * 255.0)
                
                # Enhanced filtering for electronic devices
                # Using multiple characteristics to distinguish from lights and detect hidden devices
                is_device = (
                    aspect_ratio >= 0.2 and aspect_ratio <= 4.0 and  # Relaxed aspect ratio for various devices
                    len(approx) >= 3 and len(approx) <= 20 and  # Relaxed corner constraints
                    solidity >= 0.35 and  # Lowered solidity threshold for partially hidden devices
                    rectangularity >= 0.25 and  # Lowered rectangularity for angled views
                    color_variance > 8 and  # Lowered threshold to detect dimmer devices
                    edge_density > 0.02 and  # Lowered edge density for obscured devices
                    texture_score > 2  # Lowered texture threshold for smooth surfaces
                )
                
                # Specific criteria for phones and cameras
                is_phone = (
                    aspect_ratio >= 0.4 and aspect_ratio <= 2.2 and  # Common phone aspect ratios
                    rectangularity >= 0.6  # Phones are typically rectangular
                )
                
                is_camera = (
                    aspect_ratio >= 0.8 and aspect_ratio <= 1.2 and  # Cameras are often square-ish
                    solidity >= 0.4 and  # Cameras have distinct solid shapes
                    edge_density > 0.04  # Cameras have strong edge features
                )
                
                # Adjusted filter for spectacles
                is_spectacles = (
                    aspect_ratio >= 1.8 and
                    solidity < 0.8 and
                    rectangularity < 0.7
                )
                
                if (is_device and not is_spectacles) or is_phone or is_camera:
                    # Calculate approximate distance based on object size
                    # Assuming a typical device width of 15cm, we can estimate distance
                    # using the ratio of actual size to perceived size
                    TYPICAL_DEVICE_WIDTH = 15  # cm
                    FOCAL_LENGTH = 500  # approximate focal length for typical webcam
                    distance_cm = (TYPICAL_DEVICE_WIDTH * FOCAL_LENGTH) / w
                    distance_meters = distance_cm / 100
                    
                    # Determine if device is directly visible or hidden based on distance
                    is_visible = distance_meters <= 3  # Visible if within 3 meters
                    
                    # Choose color based on visibility (green for visible, red for hidden)
                    color = (0, 255, 0) if is_visible else (0, 0, 255)
                    
                    # Draw circle around detected object
                    center = (x + w//2, y + h//2)
                    radius = int(max(w, h) / 2)
                    cv2.circle(frame, center, radius, color, 2)
                    
                    # Add detected device to list with visibility status
                    status = "Visible" if is_visible else "Hidden"
                    detected_devices.append(f"{status} Device at ({x}, {y}) - {distance_meters:.1f}m")
                    
                    # Add label with distance
                    cv2.putText(frame, f"{status} Device ({distance_meters:.1f}m)", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, detected_devices
        
    def detect_smartphones_rf(self):
        """Detect smartphones using RF sensor"""
        # TODO: Implement actual RF sensor reading and processing
        # This would interface with RF sensor hardware to detect nearby devices
        return []
        
    def run(self):
        """Main run loop for the electronics tracker"""
        try:
            self.initialize_camera()
            self.initialize_rf_sensor()
            
            while True:
                # Camera-based detection
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                # Flip the frame horizontally (180 degrees)
                frame = cv2.flip(frame, 1)
                    
                # Process frame for device detection
                processed_frame, visual_devices = self.detect_devices_vision(frame)
                
                # RF-based smartphone detection
                rf_devices = self.detect_smartphones_rf()
                
                # Update detected devices
                current_devices = set(visual_devices + rf_devices)
                new_devices = current_devices - self.detected_devices
                if new_devices:
                    print(f"New devices detected: {new_devices}")
                self.detected_devices = current_devices
                
                # Display the processed frame
                cv2.imshow('Electronics Tracker', processed_frame)
                
                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ElectronicsTracker()
    try:
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")