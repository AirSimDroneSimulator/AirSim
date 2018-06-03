import os
import tempfile
import pprint
import time
import math

from AirSimClient import *

def headToDes(x,y):
    pos = client.getPosition()
    pos_x = client.getPosition().x_val
    pos_y = client.getPosition().y_val
    angle = math.atan((y-pos_y)/(x-pos_x))
    if x - pos_x < 0:
        if y - pos_y <0:
            angle -= math.pi
        else:
            angle += math.pi 
    return angle

def getDepth(x,y):
    depthInfo = client.simGetImages(
        [ImageRequest(1, AirSimImageType.DepthPerspective, True)]
        ) #depth in perspective projection
    imageFloat = depthInfo[0].image_data_float
    imageFloatLineList=[]
    
    for i in range(0, len(imageFloat), 256):
        imageFloatLineList.append(imageFloat[i:i+256])
    
    d = (imageFloatLineList[y][x]+imageFloatLineList[y+1][x]
        +imageFloatLineList[y][x+1]+imageFloatLineList[y+1][x+1]
        )/4
    return d

def droneAction(action):
    v = 5
    v_x = v*math.cos(angle)
    v_y = v*math.sin(angle)

    if action == 'urg':
        client.moveByVelocity(0,0,-3,1)
    elif action == 'up':
        client.moveByVelocity(v_x,v_y,-5,1)
    elif action == 'down':
        client.moveByVelocity(v_x,v_y,3,1)

    elif action == 'left':
        client.moveByVelocity((v_x/2+v_y),(v_y/2-v_x),0,3)
    elif action == 'right':
        client.moveByVelocity((v_x/2-v_y),(v_y/2+v_x),0,3)
    else:
        client.moveByVelocity(v_x,v_y,0,1)

# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

print('taking off')
client.moveToPosition(0,0,-3,5)
index = 0

while True:

    x_des = float(input('X: '))
    y_des = float(input('Y: '))
    z_des = float(input('Z: '))
    dist_2d = 9999

    if x_des == 9999:
        break

    angle = headToDes(x_des,y_des)
    client.moveByAngle(0,0,-5,angle,3)
    time.sleep(3)

    while dist_2d>5:

        pos_x = client.getPosition().x_val
        pos_y = client.getPosition().y_val
        dist_2d = math.sqrt(pow((x_des-pos_x),2)+pow((y_des-pos_y),2))
        
        d_u = getDepth(127,35)
        d_c = getDepth(127,71)
        d_d = getDepth(127,107)
        d_l = getDepth(91,71)
        d_r = getDepth(163,71)
        d_bottom = getDepth(127,142)
        print(d_u,'\n', d_l, d_c, d_r,'\n', d_d, '\n', d_bottom)

        if d_c < 5:
            action = 'urg'
        elif d_c > 15 and d_d > 15 and d_bottom > 20:
            action = 'down'
        elif d_c < 15:
            maxDepth = max(d_u,d_d,d_l,d_r)
            if maxDepth == d_l:
                action = 'left'
            elif maxDepth == d_r:
                action = 'right'
            elif maxDepth == d_d:
                action = 'down'
            else: 
                action = 'up'
        else:
            action = 'fw'
        
        droneAction(action)
        print(action)
        if action == 'left' or action == 'right':
            time.sleep(2)
            client.hover()
            time.sleep(3)
            angle = headToDes(x_des,y_des)
            client.moveByAngle(0,0,client.getPosition().z_val,angle,3)
            print(angle)
            time.sleep(3)



        responses = client.simGetImages([
                #ImageRequest(0, AirSimImageType.DepthVis),  #depth visualiztion image
                #ImageRequest(1, AirSimImageType.DepthPerspective, True), #depth in perspective projection
                ImageRequest(1, AirSimImageType.Scene)]) #scene vision image in png format
                #ImageRequest(1, AirSimImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
        print('Retrieved images: %d' % len(responses))

        tmp_dir = os.path.join(os.path.abspath('.'), "AirSim_sample_input")
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise
        for idx, response in enumerate(responses):
            #filename = os.path.join(tmp_dir, str(idx))
            filename = os.path.join(tmp_dir, str(index))
            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                AirSimClientBase.write_pfm(os.path.normpath(filename + '.pfm'), AirSimClientBase.getPfmArray(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                AirSimClientBase.write_file(os.path.normpath(filename + '.jpg'), response.image_data_uint8)
            else: #uncompressed array
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
                img_rgba = np.flipud(img_rgba) #original image is fliped vertically
                img_rgba[:,:,1:2] = 100 #just for fun add little bit of green in all pixels
                AirSimClientBase.write_png(os.path.normpath(filename + '.greener.png'), img_rgba) #write to png
        
        index +=1

client.hover()
time.sleep(2)
print(client.getPosition())

AirSimClientBase.wait_key('Press any key to reset to original state')
client.reset()
client.enableApiControl(False)
