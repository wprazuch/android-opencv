package com.example.nativeopencvandroidtemplate

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import androidx.core.app.ActivityCompat
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
//import com.example.nativeopencvandroidtemplate.ml.KerasSavedTextRecognizer
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
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.*
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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

fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
    val assetManager = context.assets
    val assetFileDescriptor = assetManager.openFd(filename)
    val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
    val startOffset = assetFileDescriptor.startOffset
    val declaredLength = assetFileDescriptor.declaredLength

    val buffer = inputStream.channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    return buffer
}

fun getModelInputOutputShapes(model: Interpreter): Pair<IntArray, IntArray> {
    val inputShape = model.getInputTensor(0).shape()
    val outputShape = model.getOutputTensor(0).shape()
    return Pair(inputShape, outputShape)
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

        var bitmap = BitmapFactory.decodeResource(getResources(),
            R.drawable.cropped_box_1)

        val MODEL_ASSETS_PATH = "text_recognizer.tflite"
        val assetFileDescriptor = assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.getFileDescriptor())
        val fileChannel = fileInputStream.getChannel()
        val startoffset = assetFileDescriptor.getStartOffset()
        val declaredLength = assetFileDescriptor.getDeclaredLength()
        val modelFile =  fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)

        val interpreter = Interpreter( modelFile )

        var tensorImage = TensorImage.fromBitmap(bitmap)

        val imageProcessorBuilder = ImageProcessor.Builder()
        imageProcessorBuilder.add(ResizeOp(32, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
        imageProcessorBuilder.add(NormalizeOp(0.0f, 255.0f))
        imageProcessorBuilder.add(NormalizeOp(0.694f, 0.299f))
        val imageProcessor = imageProcessorBuilder.build()

        tensorImage = imageProcessor.process(tensorImage)


        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 128, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)

        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 3968), DataType.FLOAT32)

        interpreter.run(inputFeature0.buffer, outputFeature0.buffer)

        val MODEL_POSTPROCESSOR_ASSETS_PATH = "text_recognizer_postprocessor.tflite"
        val assetFileDescriptor2 = assets.openFd(MODEL_POSTPROCESSOR_ASSETS_PATH)
        val fileInputStream2 = FileInputStream(assetFileDescriptor2.getFileDescriptor())
        val fileChannel2 = fileInputStream2.getChannel()
        val startoffset2 = assetFileDescriptor2.getStartOffset()
        val declaredLength2 = assetFileDescriptor2.getDeclaredLength()
        val modelFile2 =  fileChannel2.map(FileChannel.MapMode.READ_ONLY, startoffset2, declaredLength2)

        val interpreter2 = Interpreter( modelFile2 )


        val inputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 124), DataType.FLOAT32)
        inputFeature02.loadBuffer(outputFeature0.buffer)

        val outputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 400), DataType.UINT8)

        interpreter2.run(inputFeature02.buffer, outputFeature02.buffer)

        val outputText = String(outputFeature02.buffer.array(), Charsets.UTF_8)

        println(outputText)




    }

    companion object {

        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
    }
}
