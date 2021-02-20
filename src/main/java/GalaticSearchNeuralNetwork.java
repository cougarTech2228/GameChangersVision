import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

public class GalaticSearchNeuralNetwork {
    private static final Scalar MEAN = new Scalar(0.485, 0.456, 0.406);
    private static final Scalar STD = new Scalar(0.229, 0.224, 0.225);
    private static final double SCALE_FACTOR = 1 / 255.0;
    private Net net;

    ArrayList<String> imgLabels = new ArrayList<String>();


    public class Prediction {
        public String label;
        public double conf;
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public GalaticSearchNeuralNetwork() {
        imgLabels.add("RedA");
        imgLabels.add("RedB");
        imgLabels.add("BlueA");
        imgLabels.add("BlueB");

        System.out.println("Loading Neural Network...");
        net = Dnn.readNet("/home/pi/galactic_search.pb");
        System.out.println("Network Loaded.");
    }
    
    /**
     * Do the actual image prediction from an image
     * @param image the full-sized image matrix from the camera
     * @return The detected text label for the image
     */
    public Prediction predictLabel(Mat image) {
        Mat inputBlob = getPreprocessedImage(image);
        net.setInput(inputBlob);
        Mat classification = net.forward();
        int index = getPredictedClass(classification);
        String label = imgLabels.get(index);

        Prediction predict = new Prediction();
        predict.conf = classification.get(0, index)[0];
        predict.label = label;
        return predict;
    }

    private static Mat centerCrop(Mat inputImage) {
        int y1 = Math.round((inputImage.rows() - 224) / 2);
        int y2 = Math.round(y1 + 224);
        int x1 = Math.round((inputImage.cols() - 224) / 2);
        int x2 = Math.round(x1 + 224);

        Rect centerRect = new Rect(x1, y1, (x2 - x1), (y2 - y1));
        Mat croppedImage = new Mat(inputImage, centerRect);

        return croppedImage;
    }

    /**
     * images must be pre-processed to be in the format that MobileNetV2 expects
     * @param srcImage the full sized image matrix.
     * @return the processed image matrix
     */
    private static Mat getPreprocessedImage(Mat srcImage) {
        // this object will store the preprocessed image
        Mat image = new Mat();

        // resize input image
        Imgproc.resize(srcImage, image, new Size(256, 256));

        // create empty Mat images for float conversions
        Mat imgFloat = new Mat(image.rows(), image.cols(), CvType.CV_32FC3);

        // convert input image to float type
        image.convertTo(imgFloat, CvType.CV_32FC3, SCALE_FACTOR);

        // crop input image
        imgFloat = centerCrop(imgFloat);

        // prepare DNN input
        Mat blob = Dnn.blobFromImage(imgFloat, 1.0, /* default scalefactor */
                new Size(224, 224), /* target size */
                MEAN, /* mean */
                true, /* swapRB */
                false /* crop */
        );

        // divide on std
        Core.divide(blob, STD, blob);

        return blob;
    }

    /**
     * Preprocess an image from a saved file on disk (used for test only)
     * @param imagePath
     * @return
     */
    private static Mat getPreprocessedImage(String imagePath) {
        Mat imageRead;
        // get the image from the internal resource folder
        imageRead = Imgcodecs.imread(imagePath);
        return getPreprocessedImage(imageRead);
    }

    /**
     * Convert a matrix of label probabilites to actual text lables
     * @param classificationResult The result returned from the network prediction
     * @return The text label
     */
    private static int getPredictedClass(Mat classificationResult) {
        // obtain max prediction result
        Core.MinMaxLocResult mm = Core.minMaxLoc(classificationResult);
        double maxValIndex = mm.maxLoc.x;
        return (int)maxValIndex;
    }
}
