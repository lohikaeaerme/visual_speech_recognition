import numpy as np
import cv2 as cv
import dlib

class FaceDetector:
    

    def __init__(self):

        face_detection_model ='/Users/juliakisela/HKA/8.Semester/Thesis/talking_in_the_disco/make_dataset_number_lipreading/face_detection_yunet_2023mar.onnx' # 'make_dataset_number_lipreading/face_detection_yunet_2023mar.onnx'
        self.facial_points_model = dlib.shape_predictor('/Users/juliakisela/HKA/8.Semester/Thesis/talking_in_the_disco/make_dataset_number_lipreading/shape_predictor_68_face_landmarks.dat')

        score_threshold = 0.9
        nms_threshold = 0.3
        top_k = 5000

        self.detector = cv.FaceDetectorYN.create(
            face_detection_model,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )

        self.except_count = 0

    def face_detecting(self,img):
        #img = cv.imread('/Users/juliakisela/HKA/8.Semester/Thesis/talking_in_the_disco/face_detection/test.jpeg')
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        faces = self.detector.detect(img)
        return faces

    def cut_faces(self, img):
        faces = self.face_detecting(img)

        cropped_img = None
        padding = 30
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                x = coords[0] - padding
                y = coords[1] - padding 
                width = coords[2] + 2* padding
                height = coords[3] + 2 * padding
                
                cropped_img = img[y : y+height, x : x + width]
        #else:
        #    print('No face detected to cut out')
        return cropped_img

    def shape_to_np(self,shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords
    
    def predict_facial_points(self,face_img,resize=500):
        try:
            size_y = int((resize / face_img.shape[0]) * face_img.shape[1])
            face_img = cv.resize(face_img,dsize=(size_y,resize))
            faceBoxRectangleS = dlib.rectangle(left=0, top=0, right=500, bottom=500)
            shape = self.facial_points_model(face_img, faceBoxRectangleS)
            shape = self.shape_to_np(shape)

            return shape, face_img
        except:
            #print('Error in facial points')
            return None,None

    def cut_lips(self, image):
        try:
            img_face = self.cut_faces(image)
            if (img_face is None):
                return None
            
            size_y = int((512 / img_face.shape[0]) * img_face.shape[1])
            img = cv.resize(img_face,dsize=(size_y,512))
            faces, _ = self.face_detecting(img)
        except Exception as e:
            print("Exception in cut lips -- should not happen : " + str(self.except_count))
            self.except_count += 1
            return None
        #print(type(faces[1]))
        cropped_img = None
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                #print(f'lip left: {coords[10]}, {coords[11]}, lips right: {coords[12]}, {coords[13]}')
                center_point_x = int((coords[10] + coords[12]) / 2)
                center_point_y = int((coords[11] + coords[13]) / 2)

                half_picture_width = 112
                cropped_img = img[center_point_y-half_picture_width:center_point_y+half_picture_width, center_point_x-half_picture_width:center_point_x+half_picture_width]
                #cv.imwrite('/Users/juliakisela/HKA/8.Semester/Thesis/talking_in_the_disco/face_detection/result.jpeg', cropped_img)
        else:
            print('No face detected on cropped image')
        return cropped_img


    def cut_multiple_lips(self, images, padding = 40):
        try:
            min_x = 1000
            max_x = 0
            min_y = 1000
            max_y = 0
            
            for image in images:
                faces = self.face_detecting(image)
                if faces[1] is not None:
                    for idx, face in enumerate(faces[1]):
                        coords = face[:-1].astype(np.int32)
                        x = coords[0] - padding
                        y = coords[1] - padding 
                        width = coords[2] + 2* padding
                        height = coords[3] + 2 * padding

                        if min_x > x :
                            min_x = int( x )
                        if max_x < x + width:
                            max_x = int( x + width)
                        if min_y > y :
                            min_y = int( y)
                        if max_y < y + height:
                            max_y = int( y + height)
                else:
                    print('face : ' + str(faces))
                    print('this is bad')

            face_images = [] 
            print([min_y ,max_y, min_x , max_x])
            for image in images:
                cropped_img = image[min_y : max_y, min_x : max_x]
                size_y = int((512 / cropped_img.shape[0]) * cropped_img.shape[1])
                resized_droped_img = cv.resize(cropped_img,dsize=(size_y,512))
                face_images.append(resized_droped_img)
            
            lip_images = []
            for face_img in face_images:
                faces = self.face_detecting(face_img)
                if faces[1] is not None:
                    coords = faces[1][0][:-1].astype(np.int32)
                    #print(f'lip left: {coords[10]}, {coords[11]}, lips right: {coords[12]}, {coords[13]}')
                    center_point_x = int((coords[10] + coords[12]) / 2)
                    center_point_y = int((coords[11] + coords[13]) / 2)

                    half_picture_width = 112
                    cropped_img = face_img[center_point_y-half_picture_width:center_point_y+half_picture_width, center_point_x-half_picture_width:center_point_x+half_picture_width]
                    lip_images.append(cropped_img)
                else:
                    print('error Cutting lips')

            return lip_images
        except:
            return []



if __name__ == '__main__':

    ## [initialize_FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(
        face_detection_model,
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )

    faces, img = face_detecting()
    cut_lips(faces, img)