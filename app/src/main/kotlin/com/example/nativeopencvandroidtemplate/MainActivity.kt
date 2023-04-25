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
            R.drawable.cropped_box_3)

        // using mat class
        val mapped_crop = Mat()
        Utils.bitmapToMat(bitmap,mapped_crop)

        println("TESTESTEST")

//        var crop = Mat()

//        if (mapped_crop.type() == CvType.CV_8U) {
//            crop.convertTo(crop, CvType.CV_64FC3)
//            Core.divide(crop, Scalar(255.0), crop)
//        }
//        else if (mapped_crop.type() == CvType.CV_8UC4) {
//            Imgproc.cvtColor(mapped_crop, crop, Imgproc.COLOR_RGBA2RGB)
//            crop.convertTo(crop, CvType.CV_64FC3)
//            Core.divide(crop, Scalar(255.0), crop)
//        }

//        Imgproc.resize(crop, crop, outputSize, 0.0, 0.0, Imgproc.INTER_LINEAR)

// In that case we need to pad because we want to enforce both width and height
        val offset = Pair(0, 0)
//        val paddedImg = padToBoundingBox(crop, offset.first, offset.second, 32, 128)

//        val batches = listOf(paddedImg)



////        val mean = MatOfDouble(0.6940000057220459, 0.6949999928474426, 0.6930000185966492)
////        val std = MatOfDouble(0.299, 0.296, 0.301)
//
//        val width = 32
//        val height = 128
//        val depth = 3
//        val arr = Array(width) { Array(height) { DoubleArray(depth) } }
//
//        // Fill the array with unique values for each channel
//        for (i in 0 until width) {
//            for (j in 0 until height) {
//                arr[i][j][0] = 0.6940000057220459
//                arr[i][j][1] = 0.6949999928474426
//                arr[i][j][2] = 0.6930000185966492
//            }
//        }
//
//        // Create a Mat object from the array
//        val mean = Mat(width, height, CvType.CV_64FC3)
//        for (i in 0 until width) {
//            for (j in 0 until height) {
//                mean.put(i, j, *arr[i][j])
//            }
//        }
//
//        // Fill the array with unique values for each channel
//        for (i in 0 until width) {
//            for (j in 0 until height) {
//                arr[i][j][0] = 0.299
//                arr[i][j][1] = 0.296
//                arr[i][j][2] = 0.301
//            }
//        }
//
//        // Create a Mat object from the array
//        val std = Mat(width, height, CvType.CV_64FC3)
//        for (i in 0 until width) {
//            for (j in 0 until height) {
//                std.put(i, j, *arr[i][j])
//            }
//        }

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

//        Core.subtract(paddedImg, mean, paddedImg)
//        Core.divide(paddedImg, std, paddedImg)

//        Core.subtract(paddedImg, Scalar(0.694), paddedImg)
//        Core.divide(paddedImg, Scalar(0.299), paddedImg)






        println("TESTESTES")
//
//        val modelFile = "keras_saved_text_recognizer"
//        val model = Interpreter(loadModelFile(this, modelFile))
//        val (inputShape, outputShape) = getModelInputOutputShapes(model)


        val MODEL_ASSETS_PATH = "text_recognizer.tflite"
        val assetFileDescriptor = assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.getFileDescriptor())
        val fileChannel = fileInputStream.getChannel()
        val startoffset = assetFileDescriptor.getStartOffset()
        val declaredLength = assetFileDescriptor.getDeclaredLength()
        val modelFile =  fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)

        val interpreter = Interpreter( modelFile )
//        interpreter.run( inputs , outputs )

//        val model = KerasSavedTextRecognizer.newInstance(this)
        var tensorImage = TensorImage.fromBitmap(bitmap)

        val imageProcessorBuilder = ImageProcessor.Builder()
        imageProcessorBuilder.add(ResizeOp(32, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
        imageProcessorBuilder.add(NormalizeOp(0.0f, 255.0f))
        imageProcessorBuilder.add(NormalizeOp(0.694f, 0.299f))
        val imageProcessor = imageProcessorBuilder.build()

        tensorImage = imageProcessor.process(tensorImage)


        var input_ = interpreter.getInputTensor(0)
        var output_ = interpreter.getOutputTensor(0)

//        var input_2 = Tensor(tensorImage.tensorBuffer)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 128, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)

//        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 124), DataType.FLOAT32)
        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 3968), DataType.FLOAT32)

        interpreter.run(inputFeature0.buffer, outputFeature0.buffer)

//        input_ = tensorImage.tensorBuffer
        println("TESTEST")
//        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 128, 3), DataType.FLOAT32)
//        inputFeature0.loadBuffer(tensorImage.buffer)
//        val outputs = interpreter.run(inputFeature0, )
//
//        println(tensorImage.dataType)
//
//// Creates inputs for reference.
//        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 128, 3), DataType.FLOAT32)
//        inputFeature0.loadBuffer(tensorImage.buffer)
//
//// Runs model inference and gets result.
//        val outputs = model.process(inputFeature0)
//        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//// Releases model resources if no longer used.
//        model.close()


        val MODEL_POSTPROCESSOR_ASSETS_PATH = "text_recognizer_postprocessor.tflite"
        val assetFileDescriptor2 = assets.openFd(MODEL_POSTPROCESSOR_ASSETS_PATH)
        val fileInputStream2 = FileInputStream(assetFileDescriptor2.getFileDescriptor())
        val fileChannel2 = fileInputStream2.getChannel()
        val startoffset2 = assetFileDescriptor2.getStartOffset()
        val declaredLength2 = assetFileDescriptor2.getDeclaredLength()
        val modelFile2 =  fileChannel2.map(FileChannel.MapMode.READ_ONLY, startoffset2, declaredLength2)

        val interpreter2 = Interpreter( modelFile2 )

        var input_2 = interpreter2.getInputTensor(0)
        var output_2 = interpreter2.getOutputTensor(0)

        val inputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 124), DataType.FLOAT32)
        inputFeature02.loadBuffer(outputFeature0.buffer)

//        val outputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 124), DataType.FLOAT32)
//        val outputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 1), DataType.STRING)

        val outputFeature02 = TensorBuffer.createFixedSize(intArrayOf(1, 400), DataType.UINT8)

        interpreter2.run(inputFeature02.buffer, outputFeature02.buffer)

        val sb = StringBuilder()

// Iterate over the output tensor data and append it to the StringBuilder
        val outputSize2 = output_2.shape()[0].toInt()
        for (i in 0 until outputSize2) {
            sb.append("a")
            sb.append(" ")
        }

// Convert the StringBuilder object to a String and print it
        val outputString = sb.toString()
        println(outputString)

        println("TESTESTES2")




//        val tfliteModel = File(assets.open("keras_saved_text_recognizer").toString())
//        val tfliteBuffer = tfliteModel.readBytes()
//        val tfliteModelBuffer = ByteBuffer.allocateDirect(tfliteBuffer.size)
//        tfliteModelBuffer.order(ByteOrder.nativeOrder())
//        tfliteModelBuffer.put(tfliteBuffer)
//
//        val tflite = Interpreter(tfliteModelBuffer)
//
//        val inputShape = tflite.getInputTensor(0).shape()
//        val inputDataType = tflite.getInputTensor(0).dataType()
//        val inputBuffer = ByteBuffer.allocateDirect(4 * inputShape[1] * inputShape[2] * inputShape[3])
//        inputBuffer.order(ByteOrder.nativeOrder())

// Fill the input buffer with your input data
// ...

//        val inputs = arrayOf(inputBuffer)


//        val model = KerasSavedTextRecognizer.newInstance(this)
//        val bitmap2 = Bitmap.createBitmap(paddedImg.cols(), paddedImg.rows(), Bitmap.Config.RGBA_F16)
//        var tensorImage = TensorImage.fromBitmap(bitmap)
//
//        val imageProcessorBuilder = ImageProcessor.Builder()
//        imageProcessorBuilder.add(ResizeOp(32, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
//        imageProcessorBuilder.add(NormalizeOp(0.0f, 255.0f))
//        imageProcessorBuilder.add(NormalizeOp(0.694f, 0.299f))
//        val imageProcessor = imageProcessorBuilder.build()
//
//
//        tensorImage = imageProcessor.process(tensorImage)
//
//// Creates inputs for reference.
//        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 128, 3), DataType.FLOAT32)
//        val bitmap2 = Bitmap.createBitmap(paddedImg.cols(), paddedImg.rows(), Bitmap.Config.RGBA_F16)
//        Utils.matToBitmap(paddedImg, bitmap2)
//        val byteBuffer = ByteBuffer.allocate(bitmap2.byteCount)
//        bitmap2.copyPixelsToBuffer(byteBuffer)
//        val pixels = ByteArray(paddedImg.total().toInt() * paddedImg.channels())
//        paddedImg.get(0, 0, pixels)
//        val byteBuffer = ByteBuffer.wrap(pixels)
//        val matOfDouble = MatOfDouble()
//        paddedImg.convertTo(matOfDouble, CvType.CV_64FC3)
//        val pixelValues = matOfDouble.toArray()

//        val numElements = paddedImg.total().toInt() * paddedImg.channels()
//        val floatBuffer = FloatArray(numElements)
//        paddedImg.convertTo(paddedImg, CvType.CV_32FC1)
//        paddedImg.get(0, 0, floatBuffer)

//        val byteOrder = ByteOrder.nativeOrder()
//// Calculate the number of pixels in the image
//        val pixelCount = paddedImg.width() * paddedImg.height()
//// Allocate enough space for the pixel data
//        val bufferSize = pixelCount * 3 // assuming 3 channels (RGB)
//        val buffer = ByteBuffer.allocateDirect(bufferSize)
//        buffer.order(byteOrder)
//
//        paddedImg.get(0, 0, pixelValues)


//        inputFeature0.loadBuffer(tensorImage.tensorBuffer.buffer, intArrayOf(1, 32, 128, 3))
//
//// Runs model inference and gets result.
//        val outputs = model.process(inputFeature0)
//        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//        model.close()


// Releases model resources if no longer used.

//        val text_recognizer_path = ""
//        val loaded_recognizer_file = File(text_recognizer_path)
//        val recognizerInterpreterOptions = Interpreter.Options()
//
////        val modelFile = File("path/to/your/tflite/model/file")
////        val interpreterOptions = Interpreter.Options()
////        val interpreter = Interpreter(modelFile, interpreterOptions)
//
//        val tfLiteModel = Interpreter(loaded_recognizer_file, recognizerInterpreterOptions)
//        val inputDetails = tfLiteModel.getInputTensor(0).tensorShape
//        val outputDetails = tfLiteModel.getOutputTensor(0).tensorShape
//        tfLiteModel.setTensor(inputDetails[0].index, batches)
//        tfLiteModel.run()
//        val outputData = tfLiteModel.getOutputTensor(0).buffer
//
//        val postprocessorTflite = Interpreter(File(TEXT_RECOGNIZER_POSTPROCESSOR_DOCTR_TFLITE_PATH))
//        val inputDetails = postprocessorTflite.getInputTensor(0).tensorShape
//        val outputDetails = postprocessorTflite.getOutputTensor(0).tensorShape
//        postprocessorTflite.setTensor(inputDetails[0].index, outputData)
//        postprocessorTflite.run()
//        val output = postprocessorTflite.getOutputTensor(0).buffer
    }

    companion object {

        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
    }
}
