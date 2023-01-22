package com.example.facerecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.TrainData;
import org.opencv.objdetect.CascadeClassifier;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    JavaCameraView javaCameraView;
    private CascadeClassifier faceDetector;
    private Mat previousFace;

    private LoaderCallbackInterface initCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    InputStream inputStream = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                    File cascadeClassifier = getDir("cascade", Context.MODE_PRIVATE);
                    File lbpClassifier = new File(cascadeClassifier, "lbpcascade_frontalface.xml");
                    FileOutputStream fos = null;

                    try {
                        fos = new FileOutputStream(lbpClassifier);
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                        inputStream.close();
                        fos.close();

                        faceDetector = new CascadeClassifier(lbpClassifier.getAbsolutePath());

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    try {
                        train();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    javaCameraView.enableView();
                    break;
                }
                default:
                    super.onManagerConnected(status);
            }
        }
    };



    private Mat matRGB;
    private Mat matGrey;
    private int cameraId;

    private void train() throws IOException {
        Resources res = getResources();
        int resourceId = R.raw.training_img;
        InputStream in = res.openRawResource(resourceId);
        Bitmap bitmap = BitmapFactory.decodeStream(in);
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        Bitmap bmp = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        Mat mat = new Mat();
        Utils.bitmapToMat(bmp, mat);
        previousFace = mat;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = findViewById(R.id.javaCameraView);

        if(!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, initCallback);
        }
        else
            initCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        javaCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matRGB = new Mat();
        matGrey = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        matRGB.release();
        matGrey.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        matRGB = inputFrame.rgba();
        matGrey = inputFrame.gray();
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(matRGB, faces);
        // draw bounding boxes around the faces
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(matRGB, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }

        if (previousFace != null && facesArray.length > 0) {
            // resize the current frame image to match the size of the previous face image
            Mat currentFace = new Mat();
            Imgproc.resize(matRGB.submat(facesArray[0]), currentFace, previousFace.size());

            // convert both images to the same type and depth
            Mat previousFace32F = new Mat();
            Mat currentFace32F = new Mat();
            previousFace.convertTo(previousFace32F, CvType.CV_32F);
            currentFace.convertTo(currentFace32F, CvType.CV_32F);

            // compare the images
            double faceMatch = Imgproc.compareHist(previousFace32F, currentFace32F, Imgproc.CV_COMP_CORREL);

            if (faceMatch > 0.8) {
                runOnUiThread(new Runnable() {
                    public void run() {
                        Toast.makeText(MainActivity.this, "Face match found", Toast.LENGTH_SHORT).show();
                    }
                });
            } else {
                runOnUiThread(new Runnable() {
                    public void run() {
                        Toast.makeText(MainActivity.this, "Face not matched", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }

        return matRGB;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if(item.getItemId() == R.id.swap)
            swapCamera();
        return super.onOptionsItemSelected(item);
    }

    private void swapCamera() {
        cameraId = cameraId^1;
        javaCameraView.disableView();
        javaCameraView.setCameraIndex(cameraId);
        javaCameraView.enableView();
    }
}