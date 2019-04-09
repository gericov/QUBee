from dronekit import *
from pymavlink import mavutil
import time
from math import *
import cv2
import numpy as np
import imutils.video
from picamera.array import PiRGBArray
from picamera import PiCamera

def gpsDistance(lat):
  #Returns approximate distance in meters of 1 degree of latitude and longitude at certain latitude, correct to a few centimetres, using WGS84 spheroid model
  latLength = 111132.92 - 559.82*cos(radians(lat)) + 1.175*cos(4*radians(lat)) - 0.0023*cos(6*radians(lat))
  longLength = 111412.84*cos(radians(lat)) - 93.5*cos(3*radians(lat)) + 0.118*cos(5*radians(lat))

  return [latLength, longLength]
  
def targetLocation(targetPos, heading, lat, long, alt):
  #Gets gps coordinates of target
  #Takes a targetPos array of the centre of the target's position in frame [y,x]
  #takes heading and 3d position of plane at time of image capture
  #outputs array with [lat, lon] of target
  
  #get image dimensions and centre of image for aircraft position
  imgDimensions = [480, 640]
  imgCentre = [imgDimensions[0]/2, imgDimensions[1]/2]

  #Create array for position of plane in 3D space, [lat, long, altitude]
  plane3DPos = [lat, long, alt]

  #conversion between pixels and actual distance in meters, based on height and lens
  pixToMeters = alt/520.8 #meters in 1 pixel in image at altitude during image capture

  #Get difference between image centre and target image y,x axis
  targetFromPlane = []
  for i in range(2):
    targetFromPlane.append(abs(targetPos[i] - imgCentre[i]))

  #get (b) 'bearing' of target relative to plane's heading vector
  if targetPos[0] > imgCentre[0]:
    if targetPos[1] > imgCentre[1]:
      b = degrees(atan(targetFromPlane[0]/targetFromPlane[1])) + 90
    elif targetPos[1] < imgCentre[1]:
      b = degrees(atan(targetFromPlane[1]/targetFromPlane[0])) + 180
  if targetPos[0] < imgCentre[0]:
    if targetPos[1] > imgCentre[1]:
      b = degrees(atan(targetFromPlane[1]/targetFromPlane[0]))
    elif targetPos[1] < imgCentre[1]:
      b = degrees(atan(targetFromPlane[0]/targetFromPlane[1])) + 270

  #Get bearing of target from aircraft (relative to North)
  if b < 360 - heading:
    targetBearing = b + heading
  elif b > 360 - heading:
    targetBearing = (b + heading - 360) / 2

  #Get distance of target from aircraft in meters, in y,x frame components
  distanceComp = []
  for i in range(2):
    distanceComp.append(targetFromPlane[i] * pixToMeters)
  distanceFromTarget = sqrt(distanceComp[1]**2 + distanceComp[0]**2)

  #Get components of distance to target in m in Lat and Long axis
  alignedDist = [distanceFromTarget*cos(radians(targetBearing)), distanceFromTarget*sin(radians(targetBearing))]

  #Get distance in meters of 1 degree of lat and long at image capture altitude
  gpsDist = gpsDistance(plane3DPos[0])

  #Get distance between target and aircraft in degrees latitude and longitude
  gpsTargetOffset = []
  for i in range(2):
    gpsTargetOffset.append(alignedDist[i]/gpsDist[i])

  #Get gps coords of target
  targetCoords = []
  for i in range(2):
    targetCoords.append(plane3DPos[i] + gpsTargetOffset[i])

  return targetCoords
  
def detection(route):
  print('Starting detection')
  
  #Initialising stuff
  counter = 0
  positions = []
  headings = []
  centres = []
  start = time.time()
  cmds = vehicle.commands

  camera = PiCamera()
  camera.resolution =(640, 480)
  camera.framerate = 32
  cap = PiRGBArray(camera, size=(640, 480))

  time.sleep(0.1)  # allows the camera to start-up
  print('Camera on')
  #Run detection script while still going through waypoints
  for image in camera.capture_continuous(cap, format="bgr", use_video_port=True):
    # Check if still doing navigation phase of mission
    if cmds.next == cmds.count:
      #break out of for loop if at last waypoint number (dummy waypoint) and shut off camera
      camera.close()
      print('Camera off')
      break
    
    #For flight monitoring and target location
    position = vehicle.location.global_relative_frame
    heading = vehicle.heading
    
    #Monitor flight status and print to console every 2s, or every second if close to wp
    now = time.time()
    planePos = [position.lat, position.lon, position.alt]
    dist = distanceBetween(planePos, route[cmds.next -2])
    #If close to wp, print status every second
    if dist < vehicle.airspeed * 2:
      if now - start > 1:
        print('Distance to next waypoint: %fm' % dist)
        print('Airspeed: %fm/s' % vehicle.airspeed)
        print('Altitude: %fm' % planePos[2])
        print('Heading: %fdegrees' % heading)
        print('Battery remaining: %fpercent' % vehicle.battery.level)
        print()
        #reset timer
        start = time.time()
    #if not close, print every 2s
    else:
      if now - start > 2:
        print('Distance to next waypoint: %fm' % dist)
        print('Airspeed: %fm/s' % vehicle.airspeed)
        print('Altitude: %fm' % planePos[2])
        print('Heading: %fdegrees' % heading)
        print('Battery remaining: %fpercent' % vehicle.battery.level)
        print()
        #reset timer
        start = time.time()
    
    frame = image.array

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts to gray
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # blur the gray image for better edge detection
    edged = cv2.Canny(blurred, 50, 10)  # the lower the value the more detailed it would be 

    # find contours in the thresholded image and initialize the
    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # grabs contours

    # shape detector
    for c in contours:
      peri = cv2.arcLength(c, True)  # grabs the contours of each points to complete a shape
      # get the approx. points of the actual edges of the corners
      approx = cv2.approxPolyDP(c, 0.01 * peri, True)
      if 4 <= len(approx) <= 6:
        (x, y, w, h) = cv2.boundingRect(approx) # gets the (x,y) of the top left of the square and the (w,h)
        aspectRatio = w / float(h)  # gets the aspect ratio of the width to height
        area = cv2.contourArea(c)   # grabs the area of the completed square
        hullArea = cv2.contourArea(cv2.convexHull(c))
        solidity = area / float(hullArea)
        keepDims = w > 25 and h > 25
        keepSolidity = solidity > 0.9  # to check if it's near to be an area of a square
        keepAspectRatio = 0.8 <= aspectRatio <= 1.2
        if keepDims and keepSolidity and keepAspectRatio:   # checks if the values are true
          # captures the regoin of interest with a 5 pixel wider in all 2D directions
          roi = frame[y - 5:y + h + 5, x - 5:x + w + 5]
          # centre detection
          centre = [y+h/2, x+w/2]

          #The image will go through the same procesdure to detect the four corners and rotate the picture
          grayn = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
          blurredn = cv2.GaussianBlur(grayn, (5, 5), 0)
          edgedn = cv2.Canny(blurredn, 50, 100)
          (contours, _) = cv2.findContours(edgedn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          for cn in contours:
            perin = cv2.arcLength(cn, True)
            approxn = cv2.approxPolyDP(cn, 0.01 * perin, True)
            if 4 <= len(approxn) <= 6:
              (x, y, w, h) = cv2.boundingRect(approxn)
              aspectRatio = w / float(h)
              keepAspectRatio = 0.8 <= aspectRatio <= 1.2
              if keepAspectRatio:
                angle = cv2.minAreaRect(approxn)[-1]  # -1 is the angle the rectangle is at

                if angle == 0:
                  angle_n = angle
                if -45 > angle:
                  angle_n = -(90 + angle)
                else:
                  angle_n = -angle

                rotated = imutils.rotate_bound(roi, angle_n)

                # Convert the image to grayscale and turn to outline of  the letter
                g_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                b_rotated = cv2.GaussianBlur(g_rotated, (5, 5), 0)
                t_rotated = cv2.adaptiveThreshold(b_rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 0)
                kernel = np.ones((4, 4), np.uint8)
                ee = cv2.morphologyEx(t_rotated, cv2.MORPH_CLOSE, kernel)
                e_rotated = cv2.Canny(b_rotated, 50, 100)

                # uses the outline to detect the corners for the cropping of the image
                (contours, _) = cv2.findContours(e_rotated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                for cny in contours:
                  periny = cv2.arcLength(cny, True)
                  approxny = cv2.approxPolyDP(cny, 0.01 * perin, True)
                  if 4 <= len(approxny) <= 6:
                    (xx, yy), (ww, hh), angle = cv2.minAreaRect(approxny)
                    aspectRatio = ww / float(hh)
                    keepAspectRatio = 0.8 <= aspectRatio <= 1.2
                    keep_angle = angle == 0, 90, 180, 270, 360
                    if keepAspectRatio and keep_angle:
                      (xxx, yyy, www, hhh) = cv2.boundingRect(approxny)
                      nnc_roi = ee[yyy:yyy + hhh, xxx:xxx + www]
                      #Make an array of relevant variables for target location
                      positions.append([position.lat, position.lon, position.alt])
                      headings.append(heading)
                      centres.append(centre)
                      #keep count of number of saved images
                      counter = counter +1
                      cv2.imwrite("rotated%d.png" % counter, nnc_roi)
                      print("Detected and saved a target")
                          
                          

              

              
    cap.truncate(0)
          
  print("Navigation and detection done")
  return counter, positions, headings, centres


def recognition(counter):
    print('Starting recognition thread')
    
    guesses = [0] * 35  # create a list of 35 lists
    for i in range(counter):
      img = cv2.imread("rotated%d.png" % (i+1))
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # set heights and width to be able to read the image when comparing to flatten images
      h = 30
      w = 20

      resize = cv2.resize(gray, (w, h))  # resizes the images
      nparesize = resize.reshape(1, w * h).astype(np.float32)  # changes into 1D array

      knn = cv2.ml.KNearest_create()  # initalise the knn
      # joins the train data with the train_labels
      knn.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
      # looks for the 3 nearest neighbours comparing to the flatten images (k = neighbours)
      retval, npaResults, neigh_resp, dists = knn.findNearest(nparesize, k=3)

      #current guess
      gg = int(npaResults[0][0])
      #Tranform guess in ASCII format into range 0-35
      if 48 <= gg <= 57:
        guesses[gg - 48] += 1
      elif 65 <= gg <= 90:
        guesses[gg - 57] += 1

    #find modal character guess
    #Initialise mode and prev variables for first loop through
    mode = 0
    prev = guesses[0]
    for j in range(35):
      new = guesses[j]
      if new > prev:
        prev = guesses[j]
        mode = j
    #Transform back into ASCII 
    if 0 <= mode <= 8:
      mode = mode + 48
    elif 9 <= mode <= 34:
      mode = mode + 57

    return chr(mode)

def distanceBetween(loc1, loc2):
  #Gets distance between 2 locations, takes location in array form [lat, lon, alt]

  #get conversion factor between degrees lat and long to meters
  gpsDist = gpsDistance(vehicle.location.global_relative_frame.lat)

  #Gets the 3d distance in meters between 2 gps coordinates
  displacements = []
  for i in range(3):
    displacements.append(loc1[i] - loc2[i])
    
  #Convert degrees displacements into meters
  for i in range(2):
    displacements[i] = displacements[i] * gpsDist[i]
    
  distance = sqrt(displacements[0]**2 + displacements[1]**2 + displacements[2]**2)
  return distance
            
            
def fly(route, speed, accuracy, first):
  #Flies waypoint route, to be used during navigation and speed lap(s)
  
  #Loiter while commands are uploading, also need to switch out of auto and back to properly run mission
  vehicle.mode = VehicleMode("LOITER")
 
  cmds = vehicle.commands
  #Compiles list of commands to make the aircraft navigate a waypoint list at a given accuracy (goes at default speed in pixhawk parameters), upload to pixhawk to run
  cmds.clear()
  
  #add takeoff command if first pass, will be ignored if already in the air, for some reason is needed or else will skip next waypoint
  if first == True:
    cmds.add(Command( 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0, 10))
  
  #Add commands to go to buffer points for line-up
  cmds.add(Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,0, 0, accuracy, 0, 0, route[0][0], route[0][1], route[0][2]))
  #Change speed
  cmds.add(Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,0,0, 0, speed, 0,0,0,0,0))
  cmds.add(Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,0, 0, accuracy, 0, 0, route[1][0], route[1][1], route[1][2]))
  
  for i in range(len(route)-2):
    cmds.add(Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,0, 0, accuracy, 0, 0, route[i+2][0], route[i+2][1], route[i+2][2]))
  
  #Add dummy waypoint, for monitoring the end of the navigation phase
  cmds.add(Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,0, 0, accuracy, 0, 0, route[5][0], route[5][1], route[5][2]))
  
  #Upload commands and switch plane to auto mode to start uploaded mission
  cmds.upload()
  print('Commands uploaded!')
  vehicle.mode = VehicleMode("AUTO")
      
def payloadDrop(target, height):
  #Loiter while commands are uploading, also need to switch out of auto and back to properly run mission
  #vehicle.mode = VehicleMode("LOITER")
  print('Going to drop site')
  
  cmds = vehicle.commands
  # Upload command to go to waypoint
  cmds.clear()
  cmds.add(Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,0, 0, 5, 0, 0, target[0], target[1], height))
  
  #Upload commands and switch plane to auto mode to start uploaded mission
  cmds.upload()
  vehicle.mode = VehicleMode("AUTO")
  
  #Servo channels and pwm values for payload release
  servoChan1 = 6
  servoChan2 = 7
  pwm1 = 300
  pwm2 = 400
  target.append(height)
  #Once close enough to waypoint, override payload servo channels to release
  while True:
    pos = vehicle.location.global_relative_frame
    planePos = [pos.lat, pos.lon, pos.alt]
    
    #Drop payload 10m? off the target coords
    #############need to calc real offset distance#######
    if distanceBetween(planePos, target) < 10:
      vehicle.channels.overrides = {str(servoChan1):pwm1, str(servoChan2):pwm2}
      print('Payload away')
      break

print('Connecting to drone...')    
    
#Connect to vehicle and print some info 
vehicle = connect('192.168.43.95:14550', wait_ready=True, baud=921600)


print('Connected to drone')
print('Autopilot Firmware version: %s' % vehicle.version)
print('Global Location: %s' % vehicle.location.global_relative_frame)

#Load training and classification data
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)

# reshape classifications array to 1D for k-nn
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

# Create waypoints
wp1 = [54.89197380, -6.17115140, 30]
wp2 = [54.89261550, -6.16954210, 30]
wp3 = [54.89402080, -6.16964130, 30]
wp4 = [54.89429380, -6.17196680, 30]

#create buffer points to line up with waypoint route correctly
buffer1 = [54.8926279, -6.1729109, 30]
buffer2 = [54.8917270, -6.1719131, 30]

# Create waypoint route
route = [buffer1, buffer2, wp1, wp2, wp3, wp4]


# Checks every second if the pixhawk has been set to auto mode by the transmitter
while vehicle.mode.name != "AUTO":
  print("Waiting for autonomous handover...")
  time.sleep(1)
  
#When in auto, let ground station now
print('Entering autonomous navigation mode')

#Do navigation of waypoints - tolerence of 12m radius - based on min turn rate with payload
fly(route, 18, 12, True)

# Run detection algorithm while aircraft is flying through waypoints the first time
(counter, positions, headings, centres) = detection(route)
time.sleep(1)
#Try and get target information
try:
  #Get middle position within lists outputted by detection function
  middle = int(len(positions)/2)

  #After navigation, calculate target gps coordinates
  targetGPS = targetLocation(centres[middle], headings[middle], positions[middle][0], positions[middle][1], positions[middle][2])
  #If successful, mark as target acquired
  print('Target acquired at: ', targetGPS)
  haveTarget = True  
  
#If the target wasn't detected and the lists are empty
except:
  print('Target not found, doing another pass')
  #Do another pass by of waypoints but slower and scan for target again
  fly(route, 15, 12, False)
  (counter, positions, headings, centres) = detection(route)
  
  #Try and process target information again
  try:
    #Get middle position within lists outputted by detection function
    middle = int(len(positions)/2)

    #After navigation, calculate target gps coordinates
    targetGPS = targetLocation(centres[middle], headings[middle], positions[middle][0], positions[middle][1], positions[middle][2])
    #If successful, mark as target acquired
    print('Target acquired at: ', targetGPS)
    haveTarget = True
  
  #If still no target, mark as no target so aircraft returns to base without doing payload drop or speed lap
  except:
    print('No target found :(')
    haveTarget = False    

#If the target has been acquired, do payload drop and speed lap  
if haveTarget == True:

  #Drop payload at target gps coords from 20m
  payloadDrop(targetGPS, 20)
  
  #Send speed laps commands to do - at 30m/s and with a 10m radius tolerance
  fly(route, 30, 10, False)
  print('Doing speed lap')

  #Perform character recognition on saved images
  character = recognition(counter)

  print("Found a %s at %f degrees latitude and %f degrees longitude" % (character, targetGPS[0], targetGPS[1]))
  cmds = vehicle.commands
  start = time.time()
  while cmds.next < cmds.count:
    pos = vehicle.location.global_relative_frame
    planePos = [pos.lat, pos.lon, pos.alt]
      
    #Monitor flight status and print to console every 2s, or every second when close to wp
    now = time.time()
    dist = distanceBetween(planePos, route[cmds.next - 2])
    if dist > vehicle.airspeed *2:
      if now - start > 1:
        print("Distance to next waypoint: %fm" % dist)
        print("Airspeed: %fm/s" % vehicle.airspeed)
        print("Altitude: %fm" % vehicle.location.global_relative_frame.alt)
        print("%f percent battery remaining" % vehicle.battery.level)
        #reset timer
        start = time.time()
    else:
      if now - start > 2:
        print("Distance to next waypoint: %fm" % dist)
        print("Airspeed: %fm/s" % vehicle.airspeed)
        print("Altitude: %fm" % vehicle.location.global_relative_frame.alt)
        print("%f percent battery remaining" % vehicle.battery.level)
        #reset timer
        start = time.time()
        
#If no target acquired, go back to base for manual landing.
else:
  print('Going back to base')

#Return to launch and loiter
vehicle.mode = VehicleMode("RTL")

print('Mission done')
print('The target is a %s at:' % character)
del targetGPS[2]
print(targetGPS)
vehicle.close()
