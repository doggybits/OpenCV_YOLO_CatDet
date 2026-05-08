# face_recognition_module.py
import cv2
import face_recognition
import os
import pickle
import numpy as np
from datetime import datetime
import threading
from collections import deque

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir="known_faces", encodings_file="face_encodings.pkl"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        
        # Optimization settings
        self.frame_skip = 3  # Process face recognition every N frames
        self.frame_counter = 0
        self.last_face_locations = []
        self.last_face_names = []
        self.resize_factor = 0.5  # Increased from 0.25 to process fewer pixels (faster)
        
        # Create directory if it doesn't exist
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
        
        # Load existing encodings
        self.load_encodings()
    
    def load_encodings(self):
        """Load saved face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"✅ Loaded {len(self.known_face_names)} known faces")
            except:
                print("⚠️ Could not load encodings file")
    
    def save_encodings(self):
        """Save face encodings to file"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        print(f"✅ Saved {len(self.known_face_names)} face encodings")
    
    def add_new_face(self, frame, face_location):
        """Add a new face to the database"""
        top, right, bottom, left = face_location
        
        # Get face encodings (use original size for better accuracy)
        face_encodings = face_recognition.face_encodings(frame, [face_location])
        
        if not face_encodings:
            print("❌ Could not encode face. Try moving closer or better lighting.")
            return False
        
        # Get name from user
        name = self.get_name_from_user()
        
        if name and name.strip():
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            self.save_encodings()
            
            # Save face image for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(self.known_faces_dir, f"{name}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame[top:bottom, left:right])
            
            print(f"✅ Added face for '{name}'")
            return True
        
        return False
    
    def get_name_from_user(self):
        """Get name input from user"""
        print("\n" + "="*50)
        print("📝 ADDING NEW FACE")
        print("="*50)
        name = input("Enter name for this face (or press Enter to cancel): ").strip()
        print("="*50 + "\n")
        return name if name else None
    
    def recognize_faces(self, frame):
        """Optimized face recognition with frame skipping"""
        self.frame_counter += 1
        
        # Skip frames for performance
        if self.frame_counter % self.frame_skip != 0:
            return self.last_face_locations, self.last_face_names
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations (using HOG model - faster than CNN)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        # Early exit if no faces
        if not face_locations:
            self.last_face_locations = []
            self.last_face_names = []
            return [], []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            if self.known_face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                if True in matches:
                    match_indices = [i for i, match in enumerate(matches) if match]
                    
                    # Get distances for better matching
                    distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(distances) if distances.size > 0 else -1
                    
                    if best_match_index >= 0 and matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
            else:
                face_names.append("Unknown")
        
        # Scale back up face locations
        scale_factor = 1 / self.resize_factor
        self.last_face_locations = [
            (int(top * scale_factor), int(right * scale_factor), 
             int(bottom * scale_factor), int(left * scale_factor)) 
            for (top, right, bottom, left) in face_locations
        ]
        self.last_face_names = face_names
        
        return self.last_face_locations, self.last_face_names

    def draw_face_results(self, frame, face_locations, face_names):
        """Draw bounding boxes and names on frame"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            label_bg = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (left, top - label_bg[1] - 5), (left + label_bg[0] + 10, top), color, -1)
            
            # Draw name
            cv2.putText(frame, name, (left + 5, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

    def add_face_interactive(self, frame):
        """Interactive mode to add a face"""
        # Detect faces (optimized)
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        if not face_locations:
            print("❌ No face detected. Make sure your face is visible.")
            return False
        
        # Scale back up
        scale_factor = 1 / self.resize_factor
        face_locations = [
            (int(top * scale_factor), int(right * scale_factor), 
             int(bottom * scale_factor), int(left * scale_factor)) 
            for (top, right, bottom, left) in face_locations
        ]
        
        # Use the first face detected
        if len(face_locations) > 1:
            print(f"⚠️ {len(face_locations)} faces detected. Using the first one.")
        
        return self.add_new_face(frame, face_locations[0])