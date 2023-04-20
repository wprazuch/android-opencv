package com.example.nativeopencvandroidtemplate

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import androidx.core.app.ActivityCompat
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.MatOfDouble
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

fun padToBoundingBox(image: Mat, offsetHeight: Int, offsetWidth: Int, targetHeight: Int, targetWidth: Int): Mat {
    val imageHeight = image.height()
    val imageWidth = image.width()

    // Calculate the amount of padding needed for each dimension
    val padTop = maxOf(-offsetHeight, 0)
    val padBottom = maxOf(offsetHeight + targetHeight - imageHeight, 0)
    val padLeft = maxOf(-offsetWidth, 0)
    val padRight = maxOf(offsetWidth + targetWidth - imageWidth, 0)

    // Pad the image with zeros
    val paddedImage = Mat(Size((imageWidth + padLeft + padRight).toDouble(),
        (imageHeight + padTop + padBottom).toDouble()
    ), image.type(), Scalar(0.0, 0.0, 0.0))
    Core.copyMakeBorder(image, paddedImage, padTop, padBottom, padLeft, padRight, Core.BORDER_CONSTANT)

    // Crop the padded image to the target size
    val croppedImage = Mat(paddedImage, Rect(Point(padLeft.toDouble(), padTop.toDouble()), Size(targetWidth.toDouble(), targetHeight.toDouble())))

    return croppedImage
}
class MainActivity : Activity() {


    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("native-lib")

                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(
            this@MainActivity,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )

        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

        var cropPath = "/Users/wprazuch/Projects/Netguru/letsmark-ocr/data/cropped_box_2.jpg"

//        val crop = Imgcodecs.imread(getResources().getDrawable(R.drawable.cropped_box_1))

//        var bitmap = getResources().getDrawable(R.drawable.cropped_box_1)
//        var bitmap = MediaStore.Images.Media.getBitmap(R.drawable.cropped_box_1)
//        bitmap= MediaStore.Images.Media.getBitmap(this.getContentResolver(),data.getData());
//        imageView.setImageBitmap(bitmap);

        val splitWideCrops = true
        val criticalAr = 8
        val targetAr = 6
        val dilFactor = 1.4
        val outputSize = Size(32.0, 128.0)
        val preserveAspectRatio = true
        val symmetricPad = false
        val method = Imgproc.INTER_LINEAR
        val vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ£€¥¢฿"
        val beamWidth = 1
        val topPaths = 1
        var remapped = false

        var bitmap = BitmapFactory.decodeResource(getResources(),
            R.drawable.cropped_box_1)

        // using mat class
        val mapped_crop = Mat()
        Utils.bitmapToMat(bitmap,mapped_crop)

        println("TESTESTEST")

        var crop = Mat()

        if (mapped_crop.type() == CvType.CV_8U) {
            crop.convertTo(crop, CvType.CV_64FC3)
            Core.divide(crop, Scalar(255.0), crop)
        }
        else if (mapped_crop.type() == CvType.CV_8UC4) {
            Imgproc.cvtColor(mapped_crop, crop, Imgproc.COLOR_RGBA2RGB)
            crop.convertTo(crop, CvType.CV_64FC3)
            Core.divide(crop, Scalar(255.0), crop)
        }

        Imgproc.resize(crop, crop, outputSize, 0.0, 0.0, Imgproc.INTER_LINEAR)

// In that case we need to pad because we want to enforce both width and height
        val offset = Pair(0, 0)
        val paddedImg = padToBoundingBox(crop, offset.first, offset.second, 32, 128)

        val batches = listOf(paddedImg)



//        val mean = MatOfDouble(0.6940000057220459, 0.6949999928474426, 0.6930000185966492)
//        val std = MatOfDouble(0.299, 0.296, 0.301)

        val width = 32
        val height = 128
        val depth = 3
        val arr = Array(width) { Array(height) { DoubleArray(depth) } }

        // Fill the array with unique values for each channel
        for (i in 0 until width) {
            for (j in 0 until height) {
                arr[i][j][0] = 0.6940000057220459
                arr[i][j][1] = 0.6949999928474426
                arr[i][j][2] = 0.6930000185966492
            }
        }

        // Create a Mat object from the array
        val mean = Mat(width, height, CvType.CV_64FC3)
        for (i in 0 until width) {
            for (j in 0 until height) {
                mean.put(i, j, *arr[i][j])
            }
        }

        // Fill the array with unique values for each channel
        for (i in 0 until width) {
            for (j in 0 until height) {
                arr[i][j][0] = 0.299
                arr[i][j][1] = 0.296
                arr[i][j][2] = 0.301
            }
        }

        // Create a Mat object from the array
        val std = Mat(width, height, CvType.CV_64FC3)
        for (i in 0 until width) {
            for (j in 0 until height) {
                std.put(i, j, *arr[i][j])
            }
        }

//        val mean = MatOfDouble(128, 32, CvType.CV_64FC3)
//        for (i in 0 until mean.rows()) {
//            for (j in 0 until mean.cols()) {
//                val values = doubleArrayOf(1.0, 2.0, 3.0)
//                mean.put(i, j, *values)
//            }
//        }

//        val meanMat = Mat(1, 1, CvType.CV_64FC3)
//        meanMat.put(0, 0, 0.6940000057220459, 0.6949999928474426, 0.6930000185966492)
//        val newMeanMat = meanMat.reshape(3, ArrayOf(128, 3, 3))

        // create a DoubleArray with 128 copies of mean values
//        val meanArray = DoubleArray(128 * 3 * 3) { i ->
//            val index = i % 9
//            when (index) {
//                0, 1, 2 -> mean[index]
//                else -> 0.0
//            }
//        }

        // reshape the DoubleArray into a Mat object of size 128x3x3
//        val meanMat = Mat(128, 3, CvType.CV_64FC3, meanArray).reshape(3, intArrayOf(128, 3, 3))

        Core.subtract(paddedImg, mean, paddedImg)
        Core.divide(paddedImg, std, paddedImg)



        println("TESTESTES")
    }

    companion object {

        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
    }
}
