package pl.mazek27.algorytmy;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import pl.mazek27.algorytmy.Color.Colors;
import pl.mazek27.algorytmy.IO.IOfile;

import java.util.ArrayList;
import java.util.List;

public class Main {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    static String act = "p2";

    static Mat sourceImage;

    public static void main(String[] args) {
        sourceImage = IOfile.LoadPicture("C:\\photo\\source\\"+act+".jpg");
        Mat binary = preproc(sourceImage);
        Imgproc.cvtColor(binary,binary,Imgproc.COLOR_GRAY2BGR);
        contours(binary);
        IOfile.SavePicture(bitwise(sourceImage,preproc(sourceImage),contours(binary)),"C:\\photo\\output\\"+act+".jpg");
    }

    public static int[] getRange(double[] bgr){
        Mat tmp = new Mat(1,1,CvType.CV_8UC3);
        tmp.put(0,0, bgr);
        Mat color = tmp.clone();
        Imgproc.cvtColor(tmp,color, Imgproc.COLOR_BGR2HSV);
        System.out.println(tmp.toString());
        int[] range = new int[2];
        range[0] = (int)color.get(0,0)[0]-10;
        System.out.println(range[0]);
        range[1] = (int)color.get(0,0)[0]+10;
        System.out.println(range[1]);
        return range;
    }

    public static Mat preproc(Mat frame) {
        Mat tmp = frame.clone();
        Mat preprosMat = new Mat(frame.size(),CvType.CV_8UC3);
        Mat lower = frame.clone();
        Mat upper = frame.clone();
        Mat color = frame.clone();
        Imgproc.cvtColor(frame, tmp, Imgproc.COLOR_BGR2HSV);
        Core.inRange(tmp,
                new Scalar(0, 100,100),
                new Scalar(10, 255, 255),
                lower); //RED
        Core.inRange(tmp,
                new Scalar(160, 100,100),
                new Scalar(180, 255, 255),
                upper); //RED
        Core.addWeighted(lower, 1,upper, 1, 0, preprosMat );
        Core.inRange(tmp,
                new Scalar(20, 100,100),
                new Scalar(40, 255,255),
                color); //YELLOW
        Core.addWeighted(color, 1,preprosMat, 1, 0, preprosMat );
        Core.inRange(tmp,
                new Scalar(10, 100,100),
                new Scalar(18, 255,255),
                color); //ORANGE
        Core.addWeighted(color, 1,preprosMat, 1, 0, preprosMat );
        Core.inRange(tmp,
                new Scalar(45, 100,100),
                new Scalar(75, 255,255),
                color); //GREEN
        Core.addWeighted(color, 1,preprosMat, 1, 0, preprosMat );
        Core.inRange(tmp,
                new Scalar(100, 150,100),
                new Scalar(140, 255,255),
                color); //GREEN
        Core.addWeighted(color, 1,preprosMat, 1, 0, preprosMat );
        Core.inRange(tmp,
                new Scalar(10, 100,100),
                new Scalar(18, 255,255),
                color); //ORANGE
        Core.addWeighted(color, 1,preprosMat, 1, 0, preprosMat );
        Imgproc.cvtColor(frame, tmp, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(tmp, color, 127, 255, 0);
        Core.addWeighted(color, 1,preprosMat, 1, 0, preprosMat );
        Imgproc.erode(preprosMat, preprosMat, new Mat(10,10,preprosMat.type()));
        Imgproc.erode(preprosMat, preprosMat, new Mat(10,10,preprosMat.type()));
        return preprosMat;
    }

    public static List<MatOfPoint> contours(Mat frame){
        Mat gray = new Mat();
        Mat threshold = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(gray, threshold, 127,255, 0);
        ArrayList<MatOfPoint> matOfPoints = new ArrayList<>();

        Imgproc.findContours(threshold, matOfPoints, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        ArrayList<MatOfPoint> tmp = (ArrayList<MatOfPoint>) matOfPoints.clone();
        for(MatOfPoint mop : tmp){
            if(mop.toArray().length < 50){
                matOfPoints.remove(mop);
            }
        }
        System.out.println(matOfPoints.size());
        return matOfPoints;
    }

    public static Mat bitwise(Mat frame, Mat mask, List<MatOfPoint> matOfPoints){
        Mat output = new Mat(frame.size(),CvType.CV_8UC3);
        for(int y =0; y < frame.height(); y++){
            for(int x=0; x < frame.width(); x++){
                if(mask.get(x,y)[0] != 0.0){
                    output.put(x,y,Colors.getColor(Colors.bgr2hsv(frame.get(x,y))));
                }
            }
        }
        Imgproc.drawContours(output, matOfPoints, -1, new Scalar(255,0,0), 3);
        return output;
    }



}
