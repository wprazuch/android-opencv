package com.example.nativeopencvandroidtemplate

import android.Manifest
import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.core.app.ActivityCompat
import android.util.Log
import android.view.WindowManager
//import com.example.nativeopencvandroidtemplate.ml.KerasSavedTextRecognizer
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.*
import java.io.FileInputStream
import java.nio.channels.FileChannel


fun loadModel(filepath: String , context: Context): Interpreter {
    val assetFileDescriptor = context.assets.openFd(filepath)
    val fileInputStream = FileInputStream(assetFileDescriptor.getFileDescriptor())
    val fileChannel = fileInputStream.getChannel()
    val startoffset = assetFileDescriptor.getStartOffset()
    val declaredLength = assetFileDescriptor.getDeclaredLength()
    val file = fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)
    return Interpreter(file)
}

fun createImageProcessor(): ImageProcessor {
    val imageProcessorBuilder = ImageProcessor.Builder()
    imageProcessorBuilder.add(ResizeOp(32, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
    imageProcessorBuilder.add(NormalizeOp(0.0f, 255.0f))
    imageProcessorBuilder.add(NormalizeOp(0.694f, 0.299f))
    return imageProcessorBuilder.build()
}

fun preprocessBitmap(bitmap: Bitmap, imageProcessor: ImageProcessor) : TensorImage {

    var tensorImage = TensorImage.fromBitmap(bitmap)
    tensorImage = imageProcessor.process(tensorImage)
    return tensorImage
}

fun recognizeText(bitmap: Bitmap, textRecognizer: Interpreter,
                  textRecognizerPostProcessor: Interpreter,
                  imageProcessor: ImageProcessor) : String {

    var tensorImage = preprocessBitmap(bitmap, imageProcessor)
    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 128, 3), DataType.FLOAT32)
    inputFeature0.loadBuffer(tensorImage.buffer)
    val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 3968), DataType.FLOAT32)
    textRecognizer.run(inputFeature0.buffer, outputFeature0.buffer)
    val inputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 124), DataType.FLOAT32)
    inputFeature02.loadBuffer(outputFeature0.buffer)
    val outputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 400), DataType.UINT8)
    textRecognizerPostProcessor.run(inputFeature02.buffer, outputFeature02.buffer)
    val outputText = String(outputFeature02.buffer.array(), Charsets.UTF_8)
    return outputText
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

        var bitmap = BitmapFactory.decodeResource(getResources(),
            R.drawable.cropped_box_3)
        val TEXT_RECOGNIZER_ASSETS_PATH = "text_recognizer.tflite"
        val TEXT_RECOGNIZER_POSTPROCESSOR_ASSETS_PATH = "text_recognizer_postprocessor.tflite"

        val textRecognizer = loadModel(TEXT_RECOGNIZER_ASSETS_PATH, this)
        val textRecognizerPostProcessor = loadModel(TEXT_RECOGNIZER_POSTPROCESSOR_ASSETS_PATH, this)
        val imageProcessor = createImageProcessor()



        val outputText = recognizeText(bitmap, textRecognizer, textRecognizerPostProcessor, imageProcessor)

        println(outputText)

    }

    companion object {

        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
    }
}
