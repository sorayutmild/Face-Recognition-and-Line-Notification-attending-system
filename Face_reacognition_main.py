import cv2
import dlib
import numpy as np
from imutils import face_utils
import face_recognition
import pickle
import timeit
import _thread
import time
import datetime
import subprocess
import os 
import pymysql as m
from line_notify import LineNotify

ACCESS_TOKEN = "mhQirqB77N77p4heMORC9R7XUGOs17Ihtglw1r7R7xI"
notify = LineNotify(ACCESS_TOKEN)

n = 0
stream = 0 
directory='C:\\Users\\Acer\\Desktop\\face_recog_final\\program_scanface\\IMG' #ชื่อต้องเป็นภาษาอังกฤษ ไม่งีั้นหา path ไม่เจอ
known_fname_names = ["mild", "au","kao","นางญาณิกา","นางนิสราศักดิ์", "นางปารจรีย์", "นางพรพิมล", "นางพัชญ์ฐิกานต์", "นางภรภัสสรณ์", "นายสมาภณ", "นางสาวเอื้องฟ้า", "นางสาวนฤมล", "นางสาวสุกัญญา", "นางสาวอภิญญรัชต์", "นางสุภาพร", "นางอุทัย", "นางอุมาพร", "นายเจษฏา", "นายถาวร", "นายธีรเชษฐ์", "นายอัศวพงศ์","นายอุกฤต"]
# Text lname_code_email_department
known_lname_names = ["za",".l,kjhg","pui","อะสิพงศ์","เจียมเจริญศักดิ์", "ทวีธรรมวรโชติ", "พิศงาม", "เกษมสุขพัศ","ผลสุข", "จันรัมย์", "รอบคอบ", "มะโน", "คำสม", "ณาญาวิสิฏฐ์", "สว่างภพ", "จันรัมย์", "อินธิสุทธิ์", "สิงห์โต", "คำพา", "ทวีธรรมวรโชติ", "สว่างภพ", "ทีงาม"]
known_code_names = ["...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...",]
known_email_names = ["...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...","...", "...",]
known_department_names = ["kmutt","kmutt","kmutt","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW","SKW",]
Unknown_names ="Unknown"
temp = 0

# load the known faces and embeddings
print("[INFO] loading encodings...")
start_process = timeit.default_timer()
data = pickle.loads(open("encodings.pickle", "rb").read())
end_process = timeit.default_timer()
elapsed = end_process - start_process
print ("Loading my encodings that I input (Time): {0}s".format(elapsed))
face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
def print_Unknown():
    print("Name is"+ name)
    savepath = os.path.join(directory, '{}.jpg'.format(timestampname))
    image=timestampname+".jpg"
    print("Path is"+ image)
    cv2.imwrite(savepath, sub_img)
    c = None
    try:
        c = m.connect(host='localhost', user='root', passwd='123456789', db='face_recog')
        cur = c.cursor()   
        cur.execute('SET NAMES utf8;')
        sql = "INSERT INTO `face_recognition` (`fname` , `lname`, `email`, `code`, `department` ,`date_time` ,`image` ) \
                        VALUES ( '%s ','%s', '%s', '%s','%s','%s','%s') " \
                        %(Unknown_names,Unknown_names,Unknown_names,000000,Unknown_names,timestamp,image)
        sql = sql.encode('utf-8')
        try:
            cur.execute(sql)
            c.commit()
            print('เพิ่มข้อมูล เรียบร้อยแล้ว')
            print("Success")
        except:
            c.rollback()
            print('เพิ่มข้อมูล ผิดพลาด')
    except m.Error:
        print('ติดต่อฐานข้อมูลผิดพลาด')
    if c:
        c.close()
def full_step(i):
    print("Name is"+ name)
    savepath = os.path.join(directory, '{}.jpg'.format(timestampname))
    image = timestampname+".jpg"
    print("Path is"+ image)
    cv2.imwrite(savepath, sub_img)
    c = None
    try:
        c = m.connect(host='localhost', user='root', passwd='123456789', db='face_recog')
        cur = c.cursor()
        cur.execute('SET NAMES utf8;')
        sql = "INSERT INTO `face_recognition` (`fname` , `lname`, `email`, `code`, `department` ,`date_time` ,`image` ) \
                        VALUES ( '%s ','%s', '%s', '%s','%s','%s','%s') " \
                        %(known_fname_names[i],known_lname_names[i],known_email_names[i],known_code_names[i],known_department_names[i],timestamp,image)
        sql = sql.encode('utf-8')
        try:
            cur.execute(sql)
            c.commit()
            print('เพิ่มข้อมูล เรียบร้อยแล้ว')
            notify.send("แจ้งเตือน " + "\nชื่อ :"+str(known_fname_names[i])+ "\nนามสกุล :"+str(known_lname_names[i])+"\nวัน เวลา :"+timestamp, image_path=(savepath))
            print("Success")
        except:
            c.rollback()
            print('เพิ่มข้อมูล ผิดพลาด')
    except m.Error:
        print('ติดต่อฐานข้อมูลผิดพลาด')

    if c:
        c.close()

cap = cv2.VideoCapture(stream)
if not cap.isOpened():
    print("Unable to connect to camera.")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

while cap.isOpened():
    ret, frame = cap.read()
    frame1 = frame.copy()
	
    if ret:
        face_rects = detector(frame, 0)
        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            reprojectdst, euler_angle = get_head_pose(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            for start, end in line_pairs:
                cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

            cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            
            if euler_angle[1,0] >= -3 and euler_angle[1,0] <= 3 and euler_angle[0,0] >= -3 and euler_angle[0,0] <= 3 and euler_angle[2,0] >= -3 and euler_angle[2,0] <= 3:
            #if euler_angle[2,0] >= -3 and euler_angle[2,0] <= 3:
                frame = np.zeros((512,512,3), np.uint8)
                cv2.putText(frame,"Prosessing..", (20, 255), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 255), thickness=1)
                cv2.imwrite('example.png',frame1)
                img = cv2.imread("example.png")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ###NEW### Date Time FullTime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestampname = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
                ###NEW### Date Time FullTime
                  # detect the (x, y)-coordinates of the bounding boxes corresponding
                  # to each face in the input image, then compute the facial embeddings
                  # for each face
                print("[INFO] recognizing faces...")

                start_process = timeit.default_timer()
                boxes = face_recognition.face_locations(rgb,model="detection_method")
                end_process = timeit.default_timer()
                elapsed2 = end_process - start_process
                print ("Getting face location (Time): {0}s".format(elapsed2))

                start_process = timeit.default_timer()
                encodings = face_recognition.face_encodings(rgb, boxes)
                end_process = timeit.default_timer()
                elapsed3 = end_process - start_process
                print ("Getting face encodings (Time): {0}s".format(elapsed3))
                cv2.putText(img,"Getting face location (Time): {0}s".format(elapsed2), (20, frame.shape[1]-60), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), thickness=1)
                cv2.putText(img,"Getting face encodings (Time): {0}s".format(elapsed3), (20, frame.shape[1]-80), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), thickness=1)
                  # initialize the list of names for each face detected
                names = []

                start_process = timeit.default_timer()
                  # loop over the facial embeddings
                for encoding in encodings:
                    # attempt to match each face in the input image to our known
                    # encodings
                    matches = face_recognition.compare_faces(data["encodings"],
                      encoding,tolerance=0.4)
                    name = "Unknown"
                    # check to see if we have found a match
                    if True in matches:
                      # find the indexes of all matched faces then initialize a
                      # dictionary to count the total number of times each face
                      # was matched
                      matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                      counts = {}
                      # loop over the matched indexes and maintain a count for
                      # each recognized face face
                      for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                        cv2.putText(img,"Name is: {0}s".format(name), (20, frame.shape[1]-40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), thickness=1)

                      # determine the recognized face with the largest number of
                      # votes (note: in the event of an unlikely tie Python will
                      # select first entry in the dictionary)
                      name = max(counts, key=counts.get)
                    
                    # update the list of names
                    names.append(name)

                if names == [] :
                    face = 0

                end_process = timeit.default_timer()
                elapsed4 = end_process - start_process
                print ("Time taken for recognizing input image (Time): {0}s".format(elapsed4))
                  # loop over the recognized faces
                for ((top, right, bottom, left), name) in zip(boxes, names):
                    # draw the predicted face name on the image
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
                    y = top - 15 if top - 15 > 15 else top + 15
                    sub_img=frame1[top-50:bottom+50,left-50:right+50]
                    #cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 1)
                    #print(name)
                    if name == "Unknown":
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(Unknown_names), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(Unknown_names), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( print_Unknown, () )

                    if name == "mild" and temp != 1:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[0]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[0]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(0), () )
                        temp = 1

                    if name == "au" and temp != 2:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[1]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[1]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(1), () )
                        temp = 2
                        
                    if name == "kao" and temp != 3:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[2]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[2]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(2), () )
                        temp = 3

                    if name == "นางญาณิกา_อะสิพงศ์" and temp != 4:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[3]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[3]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(3), () )
                        temp = 4

                    if name == "นางนิสราศักดิ์_เจียมเจริญศักดิ์" and temp != 5:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[4]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[4]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(4), () )
                        temp = 5

                    if name == "นางปารจรีย์_ทวีธรรมวรโชติ" and temp != 6:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[5]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[5]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(5), () )
                        temp = 6

                    if name == "นางพรพิมล_พิศงาม" and temp != 7:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[6]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[6]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(6), () )
                        temp = 7

                    if name == "นางพัชญ์ฐิกานต์_เกษมสุขพัศ" and temp != 8:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[7]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[7]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(7), () )
                        temp = 8

                    if name == "นางภรภัสสรณ์_ผลสุข" and temp != 9:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[8]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[8]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(8), () )
                        temp = 9

                    if name == "นายสมาภณ_จันรัมย์" and temp != 10:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[9]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[9]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(9), () )
                        temp = 10

                    if name == "นางสาวเอื้องฟ้า_รอบคอบ" and temp != 11:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[10]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[10]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(10), () )
                        temp = 11
                    
                    if name == "นางสาวนฤมล_มะโน" and temp != 12:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[11]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[11]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(11), () )
                        temp = 12

                    if name == "นางสาวสุกัญญา_คำสม" and temp != 13:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[12]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[12]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(12), () )
                        temp = 13

                    if name == "นางสาวอภิญญรัชต์_ณาญาวิสิฏฐ์" and temp != 14:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[13]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[13]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(13), () )
                        temp = 14

                    if name == "นางสุภาพร_สว่างภพ" and temp != 15:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[14]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[14]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(14), () )
                        temp = 15

                    if name == "นางอุทัย_จันรัมย์" and temp != 16:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[15]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[15]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(15), () )
                        temp = 16

                    if name == "นางอุมาพร_อินธิสุทธิ์" and temp != 17:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[16]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[16]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(16), () )
                        temp = 17

                    if name == "นายเจษฏา_สิงห์โต" and temp != 18:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[18]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[18]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(18), () )
                        temp = 19

                    if name == "นายถาวร_คำพา" and temp != 19:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[18]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[18]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(18), () )
                        temp = 19

                    if name == "นายธีรเชษฐ์_ทวีธรรมวรโชติ" and temp != 20:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[19]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[19]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(19), () )
                        temp = 20

                    if name == "นายอัศวพงศ์_สว่างภพ" and temp != 21:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[20]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[20]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(20), () )
                        temp = 21

                    if name == "นายอุกฤต_ทีงาม" and temp != 22:
                        # Draw a label with a name below the face
                        cv2.putText(img, "CODE= "+str(known_fname_names[21]), (left,bottom+20), cv2.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1)
                        cv2.putText(img, "Name = "+str(known_code_names[21]), (left,bottom+40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1)
                        _thread.start_new_thread( full_step(21), () )
                        temp = 22

                    
                  # show the output image
                cv2.imshow("Image", img)
                cv2.imwrite('example1.png',img)
                print(temp)
        #cv2.rectangle(frame,(200,100),(400,250),(0,0,255),1)
        cv2.imshow("SKW", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
