
/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.CvSink;
import edu.wpi.cscore.VideoMode.PixelFormat;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.first.wpilibj.shuffleboard.Shuffleboard;
import edu.wpi.first.wpilibj.shuffleboard.ShuffleboardTab;
import vision.PowerTowerPipline;
import vision.GalaticSearch;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

// **************************************************************************
// * 
// * Main Class
// *
// **************************************************************************
public final class Main {
  public static final int MJPEG_OPENCV_SERVER_PORT = 1183;
  public static final double IMAGE_WIDTH_PIXELS = 640.0;
  public static final double IMAGE_HEIGHT_PIXELS = 480.0;
  public static final int DEFAULT_FRAME_RATE = 30;
  public static final double HALF_IMAGE_WIDTH_IN_PIXELS = IMAGE_WIDTH_PIXELS / 2.0;

  public static final int TARGETING_STATE_SEARCHING = 0;
  public static final int TARGETING_STATE_ACQUIRING = 1;
  public static final int TARGETING_STATE_LOCKED = 2;

  public static final double TARGET_HEIGHT_INCHES = 5.5;
  public static final double TARGET_WIDTH_INCHES = 2.0;

  public static final double TARGET_ASPECT_RATIO_TOLERANCE = .20;
  public static final double TARGET_ASPECT_RATIO_FOR_LOW_ANGLE = TARGET_HEIGHT_INCHES / TARGET_WIDTH_INCHES;
  public static final double TARGET_ASPECT_RATIO_MIN_THRESHOLD_FOR_LOW_ANGLE = TARGET_ASPECT_RATIO_FOR_LOW_ANGLE
      - (TARGET_ASPECT_RATIO_FOR_LOW_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);
  public static final double TARGET_ASPECT_RATIO_MAX_THRESHOLD_FOR_LOW_ANGLE = TARGET_ASPECT_RATIO_FOR_LOW_ANGLE
      + (TARGET_ASPECT_RATIO_FOR_LOW_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);

  // For the high angle (i.e., the ~-75 degree vision strip), we need to swap the
  // height and width in the aspect ratio calculation. TODO - not sure why this is
  // necessary
  public static final double TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE = TARGET_WIDTH_INCHES / TARGET_HEIGHT_INCHES;
  public static final double TARGET_ASPECT_RATIO_MIN_THRESHOLD_FOR_HIGH_ANGLE = TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE
      - (TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);
  public static final double TARGET_ASPECT_RATIO_MAX_THRESHOLD_FOR_HIGH_ANGLE = TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE
      + (TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);

  public static final double TARGET_LOW_ANGLE = -15.0;
  public static final double TARGET_HIGH_ANGLE = -75.0;
  public static final double TARGET_ANGLE_TOLERANCE_IN_DEGREES = 10;
  public static final double TARGET_LOW_ANGLE_MIN_THRESHOLD = TARGET_LOW_ANGLE - TARGET_ANGLE_TOLERANCE_IN_DEGREES;
  public static final double TARGET_LOW_ANGLE_MAX_THRESHOLD = TARGET_LOW_ANGLE + TARGET_ANGLE_TOLERANCE_IN_DEGREES;
  public static final double TARGET_HIGH_ANGLE_MIN_THRESHOLD = TARGET_HIGH_ANGLE - TARGET_ANGLE_TOLERANCE_IN_DEGREES;
  public static final double TARGET_HIGH_ANGLE_MAX_THRESHOLD = TARGET_HIGH_ANGLE + TARGET_ANGLE_TOLERANCE_IN_DEGREES;

  public static final int MIN_HASH_MAP_DISTANCE = 18;
  public static final int MAX_HASH_MAP_DISTANCE = 48;

  public static final double MINIMUM_HORIZONTAL_OFFSET_REQ_IN_PIXELS = 200.0;

  public static final double GS_X_OFFSET = 0;
  public static final double GS_SIZE_OFFSET = 0;

  // When we were empirically collecting data for the distance calculation hash
  // map,
  // we observed that the actual measured distance between the front of the camera
  // and the target was roughly 2 inches less than what the distance calculation
  // in
  // the code was telling us.
  public static final double DISTANCE_CORRECTION_OFFSET = 2.0;

  // !!! Very small changes in this constant dramatically affects distance calc
  // accuracy !!!
  public static final double CAMERA_FOV_ANGLE = 60.010; // FOV Angle determined empirically
  public static final double CAMERA_FOV_ANGLE_CALC = Math.tan(CAMERA_FOV_ANGLE);

  private static String configFile = "/boot/frc.json";

  public static class CameraConfig {
    public String name;
    public String path;
    public JsonObject config;
    public JsonElement streamConfig;
  }

  public static int team;
  public static boolean server;
  public static List<CameraConfig> cameraConfigs = new ArrayList<>();
  public static int targetingState = TARGETING_STATE_SEARCHING;

  // This will be the list of targets that we'll use to determine whether or not
  // we're locked on the two angle vision tape strips.
  public static List<Rect> targets = new ArrayList<>();
  public static List<RotatedRect> targetRects = new ArrayList<>();

  private static GalaticSearchNeuralNetwork galacticSearchNN = new GalaticSearchNeuralNetwork();

  private Main() {
  }

  // **************************************************************************
  // *
  // * Report parse error.
  // *
  // **************************************************************************
  public static void parseError(String str) {
    System.err.println("config error in '" + configFile + "': " + str);
  }

  // **************************************************************************
  // *
  // * Read single camera configuration
  // *
  // **************************************************************************
  public static boolean readCameraConfig(JsonObject config) {
    CameraConfig cam = new CameraConfig();

    // name
    JsonElement nameElement = config.get("name");
    if (nameElement == null) {
      parseError("could not read camera name");
      return false;
    }
    cam.name = nameElement.getAsString();

    // path
    JsonElement pathElement = config.get("path");
    if (pathElement == null) {
      parseError("camera '" + cam.name + "': could not read path");
      return false;
    }
    cam.path = pathElement.getAsString();

    // stream properties
    cam.streamConfig = config.get("stream");

    cam.config = config;

    cameraConfigs.add(cam);
    return true;
  }

  // **************************************************************************
  // *
  // * Read configuration file.
  // *
  // **************************************************************************
  public static boolean readConfig() {
    // parse file
    JsonElement top;

    try {
      top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
    } catch (IOException ex) {
      System.err.println("could not open '" + configFile + "': " + ex);
      return false;
    }

    // top level must be an object
    if (!top.isJsonObject()) {
      parseError("must be JSON object");
      return false;
    }

    JsonObject obj = top.getAsJsonObject();

    // team number
    JsonElement teamElement = obj.get("team");

    if (teamElement == null) {
      parseError("could not read team number");
      return false;
    }

    team = teamElement.getAsInt();

    // ntmode (optional)
    if (obj.has("ntmode")) {

      String str = obj.get("ntmode").getAsString();

      if ("client".equalsIgnoreCase(str)) {
        server = false;
      } else if ("server".equalsIgnoreCase(str)) {
        server = true;
      } else {
        parseError("could not understand ntmode value '" + str + "'");
      }
    }

    JsonElement camerasElement = obj.get("cameras");

    if (camerasElement == null) {
      parseError("could not read cameras");
      return false;
    }

    JsonArray cameras = camerasElement.getAsJsonArray();

    for (JsonElement camera : cameras) {
      if (!readCameraConfig(camera.getAsJsonObject())) {
        return false;
      }
    }

    return true;
  }

  // **************************************************************************
  // *
  // * Start running the camera
  // *
  // **************************************************************************
  public static VideoSource startCamera(CameraConfig config) {
    System.out.println("Starting camera '" + config.name + "' on " + config.path);
    UsbCamera camera = new UsbCamera(config.name, config.path);
    MjpegServer mjpegServer = CameraServer.getInstance().startAutomaticCapture(camera);

    Gson gson = new GsonBuilder().create();

    camera.setConfigJson(gson.toJson(config.config));
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

    if (config.streamConfig != null) {
      mjpegServer.setConfigJson(gson.toJson(config.streamConfig));
    }

    return camera;
  }

  // **************************************************************************
  // *
  // * Main Method
  // *
  // **************************************************************************
  public static NetworkTableEntry xEntryPT;
  public static NetworkTableEntry yEntryPT;
  public static NetworkTableEntry widthEntry;
  public static NetworkTableEntry heightEntry;
  public static NetworkTableEntry difference;
  public static NetworkTableEntry xOffset;
  public static NetworkTableEntry gsX;
  public static NetworkTableEntry gsY;
  public static NetworkTableEntry gsSize;
  public static NetworkTableEntry path;
  public static NetworkTableEntry neuralNetwork;
  public static  NetworkTableEntry neuralNetworkConf;
  public static VideoSource frontCamera;
  public static CvSource outputStream;
  public static Scalar greenColor;
  public static Scalar redColor;
  public static Scalar blueColor;
  public static Scalar blackColor;
  public static Scalar purpleColor;
  public static VisionThread galaticSearchThread;
  public static VisionThread powerTowerThread;

  public static void main(String... args) {
    if (args.length > 0) {
      configFile = args[0];
    }

    // Read configuration
    if (!readConfig()) {
      return;
    }

    // Start NetworkTables
    NetworkTableInstance ntinst = NetworkTableInstance.getDefault();

    if (server) {
      System.out.println("Setting up NetworkTables server");
      ntinst.startServer();
    } else {
      System.out.println("Setting up NetworkTables client for team " + team);
      ntinst.startClientTeam(team);
    }

    // Start cameras
    List<VideoSource> cameras = new ArrayList<>();

    for (CameraConfig cameraConfig : cameraConfigs) {
      cameras.add(startCamera(cameraConfig));
    }

    ShuffleboardTab tab = Shuffleboard.getTab("Power Tower");
    NetworkTableEntry xOffset = tab.add("PT Offset", 0).getEntry();

    NetworkTable table = ntinst.getTable("PowerTower");

    xEntryPT = table.getEntry("X");
    yEntryPT = table.getEntry("Y");
    widthEntry = table.getEntry("width");
    heightEntry = table.getEntry("height");
    difference = table.getEntry("difference");

    NetworkTable gsTable = ntinst.getTable("GalaticSearch");

    gsX = gsTable.getEntry("x");
    gsY = gsTable.getEntry("y");
    gsSize = gsTable.getEntry("size");
    path = gsTable.getEntry("path");
    neuralNetwork = gsTable.getEntry("Neural");
    neuralNetworkConf = gsTable.getEntry("NeuralConf");

    CvSink cvSink = new CvSink("openCV Camera");

    //Mat openCVOverlay = new Mat();

    // Start image processing on camera 0 if present
    if (cameras.size() >= 1) {

      // For OpenCV processing, you need a "source" which will be our camera and
      // a "sink" or "destination" which will be an ouputStream that is fed into 
      // an MJPEG Server. 
      // TODO - this will always get the first camera detected and that may be the back camera which is no bueno
      frontCamera = cameras.get(0);

      cvSink.setSource(frontCamera);
      outputStream = new CvSource("2228_OpenCV", PixelFormat.kMJPEG, (int) IMAGE_WIDTH_PIXELS,
          (int) IMAGE_HEIGHT_PIXELS, DEFAULT_FRAME_RATE);

      // This is MJPEG server used to create an overlaid image of what the OpenCV processing is 
      // coming up with on top of the live streamed image from the robot's front camera.
      MjpegServer mjpegServer2 = new MjpegServer("serve_openCV", MJPEG_OPENCV_SERVER_PORT);
      mjpegServer2.setSource(outputStream);

      // Just some color constants for later use in drawing contour overlays and text
      greenColor = new Scalar(0.0, 255.0, 0.0);
      redColor = new Scalar(0.0, 0.0, 255.0);
      blueColor = new Scalar(255.0, 0.0, 0.0);
      blackColor = new Scalar(0.0, 0.0, 0.0);
      purpleColor = new Scalar(255.0, 0.0, 255.0);

      galaticSearchThread = null;
      powerTowerThread = null;
      
      galaticSearchThread = makeGalacticSearch();

      powerTowerThread = makePowerTower();

    } else {
      System.out.println("No cameras found");
    }
    // **************************************************************************
    // *
    // * Main "Forever" Loop
    // *
    // **************************************************************************
    String previousSelected = null;
    Thread currentVisionThread = null;
    String visionMode = null;
    for (;;) {

      NetworkTable visionModeTable = ntinst.getTable("Vision Mode");
      NetworkTableEntry selected;
      selected = visionModeTable.getEntry("selected");
      // Start the thread's execution. Runs continuously until the program is terminated

        visionMode = selected.getString(null); 
        //System.out.println("Waiting for Shuffleboard choice... Mode: " + visionMode);

        try {
          Thread.sleep(300);
        } catch (InterruptedException ex) {
          return;
        }

        //System.out.println("Mode: " + visionMode);
          if(!visionMode.equals(previousSelected)){
            //this is the first time getting a mode
            if(previousSelected == null){
                if(visionMode.equals ("Galactic Search")){
                  galaticSearchThread.start();
                  currentVisionThread = galaticSearchThread;
                  System.out.println("Starting Galactic Search");
                }
                else if(visionMode.equals ("Power Tower")){
                  powerTowerThread.start();
                  currentVisionThread = powerTowerThread;
                  System.out.println("Starting Power Tower");
                } else {
                  visionMode = "other path we don't want";
                  currentVisionThread = null;
                }
            }
            //the mode has been changed, must destroy old thread
             else{
               try{
              currentVisionThread.stop();
              System.out.println("Stopping path");
              currentVisionThread.join();
              System.out.println("Path Stopped");
               }
               catch(Exception e){
               }
                if(visionMode.equals ("Galactic Search")){
                  galaticSearchThread = makeGalacticSearch();
                  galaticSearchThread.start();
                  System.out.println("Starting Galactic Search");
                  currentVisionThread = galaticSearchThread;
                  System.out.println("Started Galactic Search");
                }
                else if(visionMode.equals ("Power Tower")){
                  powerTowerThread = makePowerTower();
                  powerTowerThread.start();
                  System.out.println("Starting Power Tower");
                  currentVisionThread = powerTowerThread;
                  System.out.println("Started Power Tower");
                } else {
                  visionMode = "other path we don't want";
                  
                  currentVisionThread = null;
                  System.out.println("Not a valid path");
                }
            }
          }
          previousSelected = visionMode;
          //System.out.println("Vision mode: " + visionMode);
          //System.out.println("Previous mode: " + previousSelected);
        
      
        }
  
}
  private static VisionThread makeGalacticSearch(){
    return new VisionThread(frontCamera, new GalaticSearch(), pipeline -> {
      MatOfKeyPoint blobs = pipeline.findBlobsOutput();
      List<KeyPoint> keyPoints = blobs.toList();
      int num = keyPoints.size();
      //List<Number> xArray = new ArrayList<>();
      //List<Number> yArray = new ArrayList<>();
      //List<Number> sizeArray = new ArrayList<>();

      //System.out.println(num);
      Number xArray[] = new Number[num];
      Number yArray[] = new Number[num];
      Number sizeArray[] = new Number[num];

      String pathFind = "noPath";
      int minSize = 9999;
      int maxSize = 0;
      int maxGSX = 0;
      int minGSX = 9999;
      int left =0;
      int middle =0;
      int right =0;
      int i=0;
      for(KeyPoint point : keyPoints) {
        //xArray.add(point.pt.x);
        xArray[i] = point.pt.x;
        yArray[i] = point.pt.y;
        sizeArray[i] = point.size;

        if(point.size < minSize){
          minSize = (int)point.size;
        }
        if(point.size > maxSize){
          maxSize = (int)point.size;
        }
        if(point.pt.x < minGSX){
          left = i;
          minGSX = (int)point.pt.x;
        }
        if(point.pt.x > maxGSX){
          right = i;
          maxGSX = (int)point.pt.x;
        }
        i++;
      }

      if(num == 3){
        if (left == 0 && right == 2){
          middle = 1;
        }
        else if(left ==2 && right == 0){
          middle = 1;
        }
        else if(left == 0 && right == 1){
          middle = 2;
        }
        else if(left == 1 && right == 0){
          middle = 2;
        }
        else if(left == 1 && right == 2){
          middle = 0;
        }
        else if(left == 2 && right == 1){
          middle = 0;
        }
        if(maxSize - minSize >= (15 + GS_SIZE_OFFSET)){ //Red path

          if(xArray[middle].intValue() - xArray[left].intValue() > (120 + GS_X_OFFSET)){ // A path
            //System.out.println("A red path: " + (xArray[middle].intValue() - xArray[left].intValue()) + "   " + (maxSize - minSize));
            pathFind = "aRed";
          }
          else{ // B path
            //System.out.println("B red path: " + (xArray[middle].intValue() - xArray[left].intValue()) + "   " + (maxSize - minSize));
            pathFind = "bRed";
          }

        }
        else{ //Blue path

          if(xArray[right].intValue() - xArray[middle].intValue() > (120 + GS_X_OFFSET)){ // A path
            //System.out.println("A blue path: " + (xArray[middle].intValue() - xArray[left].intValue()) + "   " + (maxSize - minSize));
            pathFind = "aBlue";
          }
          else{ // B path
            //System.out.println("B blue path: " + (xArray[middle].intValue() - xArray[left].intValue()) + "   " + (maxSize - minSize));
            pathFind = "bBlue";
          }

        }
      }
      else{
        //System.out.println("sees " + num + " balls");
      }
      gsX.setNumberArray(xArray);
      gsY.setNumberArray(yArray);
      gsSize.setNumberArray(sizeArray);
      System.out.println("Setting path: " + pathFind); 
      path.setString(pathFind);
      
      Mat openCVOverlay = pipeline.cvFlipOutput();
      GalaticSearchNeuralNetwork.Prediction prediction = galacticSearchNN.predictLabel(openCVOverlay);

      outputStream.putFrame(openCVOverlay);
      neuralNetwork.setString(prediction.label);
      neuralNetworkConf.setDouble(prediction.conf);
    });
  }

  private static VisionThread makePowerTower(){
    return new VisionThread(frontCamera, new PowerTowerPipline(), pipeline -> {
      // This grabs a snapshot of the live image currently being streamed
      //cvSink.grabFrame(openCVOverlay);
      Mat openCVOverlay = pipeline.cvFlipOutput();

      double xOff = xOffset.getDouble(0.0);
      // Draw a vertical line down the center of the image (i.e., IMAGE_WIDTH / 2)
      Imgproc.line(openCVOverlay, new Point((IMAGE_HEIGHT_PIXELS / 2) + xOff, 25),
          new Point((IMAGE_HEIGHT_PIXELS / 2) + xOff, IMAGE_WIDTH_PIXELS - 10), greenColor, 3, 4);
          double greenX = (IMAGE_HEIGHT_PIXELS / 2) + xOff;

      ArrayList<MatOfPoint> convexHullsOutput = pipeline.convexHullsOutput();

      double minx = 99999;
      double miny = 99999;
      double maxx = 0;
      double maxy = 0;
      double width;
      double height;


      for (MatOfPoint points : convexHullsOutput) {
        double current_min_x = minx;
        double current_min_y = miny;
        double current_max_x = maxx;
        double current_max_y = maxy;
        boolean isValid = true;
        for (Point point : points.toArray()) {
          if(point.y < 10){
            isValid = false;
            break;
          }
          if(point.x < minx){
            minx = point.x;
          }
          if(point.x > maxx){
            maxx = point.x;
          }
          if(point.y < miny){
            miny = point.y;
          }
          if(point.y < maxy){
            maxy = point.y;
          }
        }
        if(isValid == false){
          minx = current_min_x;
          maxx = current_max_x;
          miny = current_min_y;
          maxy = current_max_y;
        }
      }
      width = maxx - minx;
      height = maxy - miny;

      //System.out.println("X: " + minx);
      xEntryPT.setDouble(minx);
      yEntryPT.setDouble(miny);
      widthEntry.setDouble(width);
      heightEntry.setDouble(height);

      // Draw a vertical line down the center of the image (i.e., IMAGE_WIDTH / 2)
      //PI CAMERA EXPOSURE NEEDS TO BE CHANGED.
      //TURN ON MANUAL EXPOSURE, AND EXPOSURE 5
      Imgproc.line(openCVOverlay,
        new Point((minx + width/2), 25),
        new Point((minx + width/2), IMAGE_WIDTH_PIXELS - 10),
        redColor, 3, 4);
        double redX = minx + width/2;
      // This overlays all of the OpenCV stuff (bounding rectangles, text, etc.) over
      // the streaming image
      outputStream.putFrame(openCVOverlay);
      difference.setDouble(greenX - redX);
    });
  }
}





  
