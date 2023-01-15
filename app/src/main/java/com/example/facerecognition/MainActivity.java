package com.example.facerecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();

    JavaCameraView javaCameraView;
    private CascadeClassifier faceDetector;

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
        for(Rect rect: faces.toArray())
        {
            Imgproc.rectangle(matRGB, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0));
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