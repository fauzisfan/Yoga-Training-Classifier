import PySimpleGUI as sg
import cv2
import numpy as np
import tensorflow as tf
import time
import argparse
import pandas as pd
import posenet
from keras.models import model_from_json, Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from sklearn import preprocessing
import joblib
import webcam_demo

"""
Demo program that displays a webcam using OpenCV
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=728)
parser.add_argument('--cam_height', type=int, default=513)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--out_data', type=str, default='pose1_data.csv')
args = parser.parse_args()

def write_csv(raw_data, outFile):
    # write the predicted into file 
    df = pd.DataFrame(raw_data,columns = ['frame',
                                          'pose_score',
                                          'part_score',
                                          'x_coord',
                                          'y_coord',
                                          'label'
							])
    df.to_csv (str(outFile), index = None, header=True)

TRAINED_MODEL_NAME = "./model/yoga_net"
scaler_file = "yoga_scaller.save"

def main(): 
    sg.theme('DarkBrown')
    
    args.file = sg.popup_get_file('Filename to play')
    args.out_data = sg.popup_get_file('Filename to save')
    
    # args.file = 'C:/Users/knum/Documents/Isfan Works/Train Data/Output data/pose1.mp4'
    # args.out_data = 'coba_gui.csv'

    # define the window layout
    layout = [[sg.Text('K-Numbers Motion Tracking Web Application', size=(60, 1), justification='Right', font='Helvetica 20'),],
              [sg.Text('Web-Camera', size=(30, 1), justification='left', font='Helvetica 20'),
               sg.Text('Trainer Motion', size=(50, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image'),
              sg.Image(filename='', key='image_vid')],
              [sg.Text('None', key='motion', size=(30, 1), justification='center', font='Helvetica 20')],
              [sg.Button('Record', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(200, 150))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(args.cam_id)
    
    recording = False

    while True:
        event, values = window.read(timeout=20)
        
        ret, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
        
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Record':
            recording = True
            
            cap = cv2.VideoCapture(args.cam_id)
            cap_vid = cv2.VideoCapture(args.file)
            
            loop = True
                
            cap.set(3, args.cam_width)
            cap.set(4, args.cam_height)
    
            start = time.time()
            frame_count = 0
             
            scores = np.empty((0,2), dtype=np.float32)
            coor = np.empty((0,2), dtype=np.float32)
            label = np.empty((0,1), dtype=np.str)
            frame = np.empty((0,1), dtype=np.int)
            keypoint = 17
            
        if recording:  
            # Reset the whole tensorflow graph
            tf.reset_default_graph()
            tf.compat.v1.global_variables_initializer()
            
            with tf.compat.v1.Session() as sess:
                with open('./model/'+'yoga_net.json', 'r') as arch_file:
                    loaded_model = model_from_json(arch_file.read())
        
                # load weights into new model
                loaded_model.load_weights(TRAINED_MODEL_NAME)
                print("Loaded model from disk")
                
                model_cfg, model_outputs = posenet.load_model(args.model, sess)
                output_stride = model_cfg['output_stride']
                
                while loop:
                    input_image, display_image, output_scale = posenet.read_cap(
                        cap, scale_factor=args.scale_factor, output_stride=output_stride)
        
                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                        model_outputs,
                        feed_dict={'image:0': input_image}
                    )
        
                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                        heatmaps_result.squeeze(axis=0),
                        offsets_result.squeeze(axis=0),
                        displacement_fwd_result.squeeze(axis=0),
                        displacement_bwd_result.squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.15)
        
                    keypoint_coords *= output_scale
        
                    # TODO this isn't particularly fast, use GL for drawing and display someday...
                    overlay_image = posenet.draw_skel_and_kp(
                        display_image, pose_scores, keypoint_scores, keypoint_coords,
                        min_pose_score=0.15, min_part_score=0.1)
                    
                    imS = cv2.resize(overlay_image, (args.cam_width, args.cam_height))
                    
                    # cv2.imshow('posenet', imS)
                    
                    imgbytes = cv2.imencode('.png', imS)[1].tobytes()  # ditto
                    window['image'].update(data=imgbytes)
                    
                    ret, frame_vid = cap_vid.read()
                    frame_out = cv2.resize(frame_vid, (args.cam_width, args.cam_height))
                    imgbytes = cv2.imencode('.png', frame_out)[1].tobytes()  # ditto
                    window['image_vid'].update(data=imgbytes)
                
                    # print(frame_count)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    for pi in range(len(pose_scores)):
                        if pose_scores[pi] <= 0.5:
                            break
                        print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                        
                        input_rt = np.empty((0,2))
                        
                        for ki, (s, c) in enumerate(zip(keypoint_scores[pi, 0:keypoint], keypoint_coords[pi, 0:keypoint, 0:keypoint])):
                            # print('Keypoint %s, score = %f, coord = [%d, %d]' % (posenet.PART_NAMES[ki], s, c[0], c[1]))
                            '''For Storing'''
                            frame = np.vstack((frame, frame_count))
                            scores = np.vstack((scores,np.array([pose_scores[pi], s])))
                            coor = np.vstack((coor,np.array([c[0], c[1]])))
                            label = np.vstack((label, "none"))
        
                            input_rt = np.vstack((input_rt,np.array([
                                # s, 
                                c[0], c[1]
                                ])))
                        
                        tempNorm = joblib.load(scaler_file)
                        
                        # normalizer = preprocessing.StandardScaler()
                        # tempNorm = normalizer.fit(input_rt.reshape(1,-1))
                        
                        # print(input_rt.reshape(1,-1).shape)
                        input_rt = tempNorm.transform(input_rt.reshape(1,-1))
                        # print(input_rt.shape)
                        
                        pred= loaded_model.predict(input_rt.reshape(1,-1))
                        val = np.argmax(pred[0], axis = 0)
                        
                        if (val==1):
                            motion = "standing"
                        elif(val==2):
                            motion = "starting"
                        elif(val==3):
                            motion = "right"
                        elif(val==4):
                            motion = "left"
                        else:
                            motion = "unknown"
                            
                        print(motion)
                        
                        window['motion'].update(motion)
                        
                    frame_count += 1
                    
                    event, values = window.read(timeout=2)
                    
                    if event == 'Stop':
                        loop = False
                        recording = False
                        
                        '''Save For Video Recorder'''
                        raw_data = np.column_stack(
                            [frame, scores, coor, label]
                            )
                        write_csv(raw_data, args.out_data)
                        
                        print('Average FPS: ', frame_count / (time.time() - start))
            
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        cap = cv2.VideoCapture(args.cam_id)
main()