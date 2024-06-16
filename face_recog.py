import os
import cv2
import face_recognition

# Function to encode known faces
def encode_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# Function to detect and recognize faces in an image
def detect_and_recognize_faces(image_path, known_face_encodings, known_face_names, output_dir):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Save the resulting image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Processed {image_path} and saved result to {output_path}")

# Main function
def main():
    known_faces_dir = "known_faces"
    test_images_dir = "test_images"
    output_dir = "output_images"

    os.makedirs(output_dir, exist_ok=True)

    known_face_encodings, known_face_names = encode_known_faces(known_faces_dir)

    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        detect_and_recognize_faces(image_path, known_face_encodings, known_face_names, output_dir)

if __name__ == "__main__":
    main()

