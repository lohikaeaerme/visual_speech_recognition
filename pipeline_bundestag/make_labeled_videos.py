import cv2 as cv
import lip_detection_and_cutout
import ffmpeg
import numpy as np
import pandas as pd
import glob
import time
from multiprocessing import Pool
import os

video_duration_in_sec = 1.5
time_before_word = 0.1
result_video_fps = 60
sequenze_frame_count = int(result_video_fps * video_duration_in_sec)

def cut_video(videopath,csvpath, video_name):

    if not os.path.exists('make_bundestag_dataset'):
        print('wrong directory')
        return

    resultpath_video = 'make_bundestag_dataset/lip_videos_words/' + video_name
    if not os.path.exists(resultpath_video):
        os.mkdir(resultpath_video)

    resultpath_lippoints = 'make_bundestag_dataset/lip_points/' + video_name
    if not os.path.exists(resultpath_lippoints):
        os.mkdir('make_bundestag_dataset/lip_points/' + video_name)

    # For ffmpeg Videos mit Ton
    resultpath_sound_videos = 'make_bundestag_dataset/face_videos_audio_words/' + video_name
    if not os.path.exists(resultpath_sound_videos):
        os.mkdir(resultpath_sound_videos)

    face_detector = lip_detection_and_cutout.FaceDetector()

    frame_shift_in_sec = 0.1

    video = cv.VideoCapture(videopath)
    fps = video.get(cv.CAP_PROP_FPS)
    print(fps)
    success,image = video.read()
    size_y = int((720 / image.shape[0]) * image.shape[1])
    image = cv.resize(image,dsize=(size_y,720))
    #image = cv.resize(image, dsize=(720,1280))

    df = pd.read_csv(csvpath)
    df.dropna()
    labels = df['Text'].to_list()
    start_times = [int(start) for start in df['Start time (ms)'].to_list()] #np.genfromtxt(csvpath,delimiter=',',missing_values='11')[:,0]

    if not len(labels) == len(start_times):
        print('error labels and time stamps not of same length : ' + str(len(labels)) + ' : '+  str(len(start_times)))
        for i in range(0,5):
            print(f'{start_times[-i]} : {labels[-i]}')
        return

    print('load csv sucessfull and first image loaded : ' + str(success))


    sucess_cutting = []

    img_array = []   
    for i in range(0,sequenze_frame_count):#int(video_duration_in_sec *fps)):
        img_array.append(image)
    

    cut_img_array = []
    sucess_cut, cut_img = cut_lip_one_img(image, cv.Mat(np.uint8(np.zeros((224, 224, 3)))),face_detector)
    for i in range(0,sequenze_frame_count):
        sucess_cutting.append(sucess_cut)
        cut_img_array.append(cut_img)

    facial_points = []
    _, facial_point = get_facial_points(image, np.zeros((68, 2)),face_detector)
    for i in range(0,sequenze_frame_count):
        facial_points.append(facial_point)
    
    # beginne mit zweitem eintrag in label-csv
    label_index = 2
    # frames read from original video
    frame_count = 0
    second_label = False
    duplicate_frames_factor = int(60 / fps) if fps != 0 else 1
    print(f'sucessfully loaded Video and csv for {videopath}')
    while success :
        success,image = video.read()
        #image = cv.resize(image, dsize=(720,1280))

        if not success:
            return
        
        size_y = int((720 / image.shape[0]) * image.shape[1])
        image = cv.resize(image,dsize=(size_y,720))

        # remove oldest frame(s)
        img_array = img_array[duplicate_frames_factor:]
        cut_img_array = cut_img_array[duplicate_frames_factor:]
        facial_points = facial_points[duplicate_frames_factor:]
        sucess_cutting = sucess_cutting[duplicate_frames_factor:]
       
        # add new frame(s)
        for i in range(0,duplicate_frames_factor):
            img_array.append(image)
            try:
                sucess_cut, cut_img = cut_lip_one_img(image,cut_img_array[-1],face_detector)
                cut_img_array.append(cut_img)
                sucess_cutting.append(sucess_cut)
                _, facial_point = get_facial_points(image,facial_points[-1],face_detector)
                facial_points.append(facial_point)

            except Exception as e:
                print('error accured in cutting Video ' + str(video_name))
                print(e)
                success = False
                break

        #print('3')
        frame_count += 1
        #print('write Video ? ' + str(frame_count % (int(fps) * frame_shift_in_sec )) + ' == 0 ')
        # alle 0.1s (alle zb 6 frames für 60fps) ein Video schreiben
        #if not allSucess(sucess_cutting):
        #    second_label + False

        if frame_count % np.ceil(fps * frame_shift_in_sec) ==  0:# and allSucess(sucess_cutting)):
            # start der aktuellen video-sequenz
            current_time = frame_count / fps - video_duration_in_sec

            # wann beginnt das nächste Label
            if len(start_times) > label_index + 1:
                next_start_time = start_times[label_index + 1]
            else:
                next_start_time = 1000000000
            label_text = None

            # mache mit erster Priorität ein zweites Video mit gleichem Label
            if(second_label):
                second_label = False
                try:
                    label_text = labels[label_index]
                except:
                    label_text = None
                    print('Error with label at index ' + str(label_index))
                    if labels[label_index] is not None:
                       print('label was : ' + str(labels[label_index]))
                    else: 
                       print('Label was None')

            # wenn current time ist in label Zeitspanne (next_start_time/label-Anfang am Anfang von video-sequenz)
            elif next_start_time < (current_time + time_before_word)*1000:
                
                
                #next_start_time bezieht sich ab jetzt auf das nachfolgende Label
                #Schleife, damit wir nicht hinter her hinken fall sehr viele Worte auf einmal kommen
                while next_start_time < (current_time + time_before_word)*1000 and len(start_times) > label_index + 1:
                    label_index += 1
                    next_start_time = start_times[label_index + 1]
                

                #if  (current_time * 1000 - start_times[label_index] > 200):
                #    print(f'label_index: {label_index} current_time: {current_time} start_time: {start_times[label_index]}')
                #    print('Just to try for now')
                #    return

                # noch ein zweites Video mit diesem Label machen
                second_label = True
                try:
                    #liest Label-text und setzt label
                    label_text = labels[label_index]     
                except:
                    label_text = None
                    print('Error with label at index ' + str(label_index))
                    if labels[label_index] is not None:
                       print('label was : ' + str(labels[label_index]))
                    else: 
                       print('Label was None')
            #print('4')
            # mache Video, wenn label = A (keine Zahl) oder Label kleiner 10 (richtiges Label)
            # kein Video, wenn mehre Zahlen gelabelt, oder kein Label in csv
            if current_time > 0:
                #cut_img_array = cut_lips(img_array)
                if allSucess(sucess_cutting) and label_text is not None and str(label_text) != 'nan' and str(label_text) != '' and str(label_text) != ' ':
                    
                    video_time = '%.1f' % current_time
                    wirteVideo(resultpath_video + '/' + video_time + '_' + str(label_text) + '.mp4'
                           ,cut_img_array,current_time,current_time + video_duration_in_sec, current_time, label_text)
                    writeTextFile(resultpath_lippoints + '/' + video_time + '_' + str(label_text) + '.txt',
                                  facial_points)
                    #writeVideoSound(resultpath_sound_videos + '/' + video_time + ':' + str(start_times[label_index]) + '_' + str(label_text) + '.mp4'
                    #       ,current_time,current_time + video_duration_in_sec, videopath)
                    
                    #wirteVideo(resultpath_video + '/' + str(frame_count) + '_' + str(label_text) + '.mp4'
                    #       ,cut_img_array,current_time,current_time + video_duration_in_sec, current_time, label_text)
                    #writeTextFile(resultpath_lippoints + '/' + str(frame_count) + '_' + str(label_text) + '.txt',
                    #              facial_points)
                    #writeVideoSound(resultpath_sound_videos + '/' + str(frame_count) + '_' + str(label_text) + '.mp4'
                    #       ,current_time,current_time + video_duration_in_sec, videopath)
                #else :
                #    print('Error wrong label ' + str(label_text) + ' at Index ' + str(label_index))

def allSucess(sucess):
    for s in sucess:
        if not s:
            return False
    return True

def cut_lip_one_img(img_to_cut,fallback_img,face_detector):
    #face, img = face_detector.face_detecting(img_to_cut)
    #new now checking for size of Face
    face = face_detector.cut_faces(img_to_cut)
    if face is not None:
        if face.shape[0] > 200:
            cropped_img = face_detector.cut_lips(img_to_cut)
            if cropped_img is not None:
                return True, cropped_img   
    
    return False, fallback_img


def get_facial_points(img_to_predict, fallback_points,face_detector):
    #face, img = face_detector.face_detecting(img_to_predict)

    face_img = face_detector.cut_faces(img_to_predict)
    if face_img is not None:
        facial_points,_ = face_detector.predict_facial_points(face_img)
        if facial_points is not None:
            return True,facial_points
        else: 
            return False, fallback_points
    else :
        return False, fallback_points


def cut_lips(img_array,face_detector):
    cropped_images = []
    for image in img_array:
        #face, img = face_detector.face_detecting(image)
        cropped_img = face_detector.cut_lips(image)
        if cropped_img is not None:
            cropped_images.append(cropped_img)    
        else:
            print('error croping image')
    return cropped_images        
               
def writeTextFile(fileName,facial_points):
    with open(fileName, "w") as file:
        for face in facial_points:
            for point in face:
                for i in point:
                    file.write(str(i) + ', ')
                    #file.write(str(point[0]) + ' , ' + str(point[1]) + ' ,')


def wirteVideo(fileName, img_array, startTime, endTime, frame_count,label):
    if(len(img_array) != sequenze_frame_count):
        print('Missing frame. img_array len ' + str(len(img_array)))
        return    

    if(startTime > 0):
        #print('writing Video ' + str(frame_count) + ' ' + str(fileName))
        image = img_array[0]
        height, width, layers = image.shape
        size = (width,height)

        out = cv.VideoWriter(fileName,cv.VideoWriter_fourcc(*'mp4v'), 60, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        #print('Done writing Video ' + str(frame_count) + ' ' + str(fileName))



def writeVideoSound(fileName, startTime, endTime, input_file):
        try:
            # Start time for trimming (HH:MM:SS)
            start_seconds = "%06.3f" % (int(startTime * 1000) % 60000 / 1000 )
            start_minutes= "%.2d" % (int(startTime/60))
            start_time = '00:' + start_minutes + ':' + start_seconds

            # End time for trimming (HH:MM:SS)
            end_seconds = "%06.3f" % (int(endTime* 1000) % 60000 / 1000 )
            end_minutes= "%.2d" % (int(endTime/60))
            end_time = '00:' + end_minutes + ':' + end_seconds

            print('start : ' + start_time)
            print('end : ' + end_time)

            (
                ffmpeg.input(input_file, ss=start_time, to=end_time)
                .output(fileName)
                .run()
            )
        except Exception as e:
            print(e)
            print('Error Write Video with Sound')

       
def make_csv(data_folder_path,csvPath,fileEnding):
    print(data_folder_path+'/*/*.'+fileEnding)
    AllVideosPaths = glob.glob(data_folder_path+'/*/*.'+fileEnding)
    labels = [path.split('_')[-1].split('.')[0] for path in AllVideosPaths]
    labels = [10 if label == "A" else label  for label in labels]
    data = {'path': AllVideosPaths, 'label': labels}
    df = pd.DataFrame(data)
    df.to_csv(csvPath+'/dataset_unbalenced.csv')

    average = int((df.count()[1] - df[df['label'] == 10].count()[1] )/ 10)
    df_no_A = df[df['label'] != 10]
    df_only_A = df[df['label'] == 10]
    df = pd.concat([df_no_A,df_only_A.sample(average,random_state=42)])
    df.to_csv(csvPath+'/dataset.csv')

def start(file):

    finished_videos = []#['micha','video','IMG_4728','IMG_0748','IMG_6562','IMG_6560','IMG_4730','IMG_6561','IMG_6563','IMG_6559','FILE022100']

    video_name = file.split('/')[-1].split('.')[0]
    file_type = file.split('.')[-1]

    if(video_name not in finished_videos):
        print('cutting Video ' + video_name)
        start = time.time()
        #try:
        cut_video(file,file.split('.')[0] + '.csv', video_name)
        #except:
        #    print('Failed with Video : ' + video_name)
        end = time.time()
        print('Done took ' + str(end - start))
    else:
        print('skipped ' + video_name)



if __name__ == '__main__':

    files = glob.glob('make_bundestag_dataset/bundestag_videos/*.mp4')
    #files = glob.glob('Datensammlung/micha.MP4')
    #start(files[0])
    error_count = 0
    with Pool(len(files)) as p:
        try:
            p.map(start,files) #Liste mit finished Videos in start
        except:
            error_count += 1
            
    print("Failed Videos : " + str(error_count))

    #p = Process(target=start, args=(files))
    #p.start()
    #p.join()

    #for file in files:
    #    video_name = file.split('/')[-1].split('.')[0]
    #    file_type = file.split('.')[-1]
    #video_name = 'micha'
    #    if(video_name not in finished_videos):
    #        try:
    #            print('cutting Video ' + video_name)
    #            start = time.time()
    #            cut_video('Datensammlung/'+video_name+ '.' + file_type,'Datensammlung/'+video_name+'.csv', video_name)
    #            end = time.time()
    #            print('Done took ' + str(end - start))
    #        except:
    #            print('Video ' + video_name + ' failed.')
    #    else:
    #        print('skipped ' + video_name)

    #TODO wieder einkommentieren wenn Videos erstellt werden
    for file in files:
        video_name = file.split('/')[-1].split('.')[0]
        num_videos = len(glob.glob('make_bundestag_dataset/lip_videos_words/' + video_name + '/*'))
        print('got ' + str(num_videos) + ' from ' + video_name)
    
    print('------------------------------------------------')
    for file in files:
        video_name = file.split('/')[-1].split('.')[0]
        num_videos = len(glob.glob('make_bundestag_dataset/lip_points/' + video_name + '/*'))
        print('got ' + str(num_videos) + ' from ' + video_name)
    
    make_csv('make_bundestag_dataset/lip_videos_words','make_bundestag_dataset/lip_videos_words','mp4')
    make_csv('make_bundestag_dataset/lip_points','make_bundestag_dataset/lip_points','txt')