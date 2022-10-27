/**
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.codelabs.objectdetection

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.io.File
import java.io.IOException
import java.math.RoundingMode
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max
import kotlin.math.min


class MainActivity : AppCompatActivity(), View.OnClickListener {
    companion object {
        const val TAG = "TFLite - ODT"
        const val REQUEST_IMAGE_CAPTURE: Int = 1
        private const val MAX_FONT_SIZE = 96F
    }

    private lateinit var captureImageFab: Button
    private lateinit var inputImageView: ImageView
    private lateinit var imgSampleOne: ImageView
    private lateinit var imgSampleTwo: ImageView
    private lateinit var imgSampleThree: ImageView
    private lateinit var tvPlaceholder: TextView
    private lateinit var currentPhotoPath: String
    private lateinit var tvResult: TextView

    // Added for UI
    private var tarWeightList = ArrayList<String>()
    private var expiryDateList = ArrayList<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        captureImageFab = findViewById(R.id.captureImageFab)
        inputImageView = findViewById(R.id.imageView)
        imgSampleOne = findViewById(R.id.imgSampleOne)
        imgSampleTwo = findViewById(R.id.imgSampleTwo)
        imgSampleThree = findViewById(R.id.imgSampleThree)
        tvPlaceholder = findViewById(R.id.tvPlaceholder)

        captureImageFab.setOnClickListener(this)
        imgSampleOne.setOnClickListener(this)
        imgSampleTwo.setOnClickListener(this)
        imgSampleThree.setOnClickListener(this)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE &&
            resultCode == Activity.RESULT_OK
        ) {
            try{
                Log.d(TAG, "ActivityResult: viewing and detecting captured image")
                setViewAndDetect(getCapturedImage())
            }catch (e : Exception){
            }
        }
    }

    /**
     * onClick(v: View?)
     *      Detect touches on the UI components
     */
    override fun onClick(v: View?) {
        Log.d(TAG, "Button Clicked.")
        when (v?.id) {
            R.id.captureImageFab -> {
                try {
                    dispatchTakePictureIntent()
                } catch (e: ActivityNotFoundException) {
                    Log.e(TAG, e.message.toString(), e)
                }
            }
            R.id.imgSampleOne -> {
                setViewAndDetect(getSampleImage(R.drawable.cyl1))
            }
            R.id.imgSampleTwo -> {
                setViewAndDetect(getSampleImage(R.drawable.cyl2))
            }
            R.id.imgSampleThree -> {
                setViewAndDetect(getSampleImage(R.drawable.cyl3))
            }
        }
    }

    /**
     * runObjectDetection(bitmap: Bitmap)
     *      TFLite Object Detection function
     */
    private fun runOpticalCharRecognition(
        bitmap: Bitmap,
        detector: ObjectDetector,
        condition: String
    ) {
        Log.d(TAG, "Running optical character recognition.")
        var scaledBitmap = Bitmap.createScaledBitmap(bitmap, bitmap.width, bitmap.height, true)
        val matrix = Matrix()
        matrix.postRotate(90F)
        for (i in 0..3) {
            scaledBitmap = Bitmap.createBitmap(
                scaledBitmap,
                0,
                0,
                scaledBitmap.width,
                scaledBitmap.height,
                matrix,
                true
            )
            //Bitmap.createScaledBitmap(bitmap, 640, 640, true)
            val croppedImage =
                TensorImage.fromBitmap(scaledBitmap)

            // Printing the final result
            val result = detector.detect(croppedImage)

            var resultValue = ""

            for (data in result) {
                val res = data.categories.toString()
                val str = res[12]

                resultValue = if (condition == "T") {
                    if (resultValue.length == 2) {  // 1 2 3 -> A 2 3
                        "${resultValue}.${str}"
                    } else {
                        "$resultValue$str"
                    }
                } else {
                    if (resultValue.isEmpty()) {
                        returnExpiryChar(str.toString())
                    } else {
                        "$resultValue$str"
                    }
                }
            }
            Log.d(TAG, "Condition is...$condition")
            Log.d(TAG, "result values is...$resultValue")

            when (condition) {
                "T" -> {
                    resultValue =
                        //roundOffDecimal(resultValue.toDouble()).toString()  //resultValue.toDouble()
                        resultValue

                       tarWeightList.add(resultValue)
                }
                "E" -> {
                    expiryDateList.add(resultValue)
                }
            }
        }
    }

    private fun roundOffDecimal(number: Double): Double {
        val df = DecimalFormat("##.#")
        df.roundingMode = RoundingMode.CEILING
        return df.format(number).toDouble()
    }

    private fun returnExpiryChar(c: String): String {
        var str = ""
        when (c) {
            "1" -> str = "A"
            "2" -> str = "B"
            "3" -> str = "C"
            "4" -> str = "D"
            "5" -> str = "E"
            "6" -> str = "F"
            "7" -> str = "G"
            "8" -> str = "H"
            "9" -> str = "I"
        }
        return str
    }

    private fun runObjectDetection(bitmap: Bitmap) {
        Log.d(TAG, "Running object detection.")

        val image = TensorImage.fromBitmap(bitmap)

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(DetParameters.maxResults)
            .setScoreThreshold(DetParameters.scoreThreshold)
            .build()

        val detector = ObjectDetector.createFromFileAndOptions(
            this,
            "tare_expiry_detection_model.tflite",
            options
        )

        val results = detector.detect(image)
        //debugPrint(results)
        val resultToDisplay = results.map {
            // Get the top-1 category and craft the display text
            val category = it.categories.first()
            val text = "${category.label}, ${category.score.times(100).toInt()}%"

            // Create a data object to display the detection result
            DetectionResult(it.boundingBox, text)
        }
        // Draw the detection result on the bitmap and show it.

        val imgWithResult = drawDetectionResult(bitmap, resultToDisplay)
        runOnUiThread {
            inputImageView.setImageBitmap(imgWithResult)
        }
        try{
            if(results.isNotEmpty()){
                val ocrOptions = ObjectDetector.ObjectDetectorOptions.builder()
                    .setMaxResults(OcrParameters.maxResults)
                    .setScoreThreshold(OcrParameters.scoreThreshold)
                    .build()

                val charDetector = ObjectDetector.createFromFileAndOptions(
                    this,
                    "digits recognition.tflite",
                    ocrOptions
                )
                // left top right bottom
                for ((i, obj) in results.withIndex()) {
                    val box = obj.boundingBox
                    val newBitmap = Bitmap.createBitmap(
                        bitmap,
                        (box.left.toInt()-50),
                        (box.top.toInt()-50),
                        (box.width().toInt()+50) ,
                        (box.height().toInt()+50)
                    )
                    Log.d(TAG,"Object detection..${i + 1}")
                    Log.d(TAG, obj.toString())
                    val res = obj.categories[0].toString()
                    //textDetection(newBitmap)

                    runOpticalCharRecognition(newBitmap, charDetector, res[11].toString())
                }
                val hashMap : HashMap<String, Int> = HashMap()
                for(i in tarWeightList){
                   if(i !in hashMap.keys){
                       hashMap[i] = 1
                   }
                    else{
                       hashMap[i] = hashMap[i]!! +1
                   }
                }
                val hashResult = hashMap.toList().sortedByDescending { (_, value) -> value}.toMap()
                tarWeightList.clear()
                var count=0
                for(i in hashResult.keys){
                    tarWeightList.add(i)
                    count+=1
                    if(count==3){
                        break
                    }
                }
                tvResult = findViewById(R.id.tv_result)

                val tvDataResult = "Tare values:-\n $tarWeightList \n Expiry date:- \n $expiryDateList"
                tvResult.text = tvDataResult
            }}
        catch (e:Exception){
            Log.e(TAG,"Error while doing object detection", e)
            throw e

        }

    }

    /**
     * setViewAndDetect(bitmap: Bitmap)
     *      Set image to view and call object detection
     */
    private fun setViewAndDetect(bitmap: Bitmap) {

        tarWeightList = ArrayList()
        expiryDateList = ArrayList()

        Log.d(TAG, "setting view and detecting.")
        // Display capture image
        inputImageView.setImageBitmap(bitmap)
        tvPlaceholder.visibility = View.INVISIBLE

        // Run ODT and display result
        // Note that we run this in the background thread to avoid blocking the app UI because
        // TFLite object detection is a synchronised process.
        // lifecycleScope.launch(Dispatchers.Default) { runObjectDetection(bitmap) }
        runObjectDetection(bitmap)
    }

    /**
     * getCapturedImage():
     *      Decodes and crops the captured image from camera.
     */
    private fun getCapturedImage(): Bitmap {
        Log.d(TAG,"Start of get captured image")

        if(!this::inputImageView.isInitialized){
            Log.d(TAG,"Input image view is null")
        }
        if(!this::currentPhotoPath.isInitialized){
            Log.d(TAG,"CURRENT Photo path is null")
        }
        Log.d(TAG, "getting captured image.")
        // Get the dimensions of the View
        try {
            val targetW: Int = inputImageView.width
            val targetH: Int = inputImageView.height

            val bmOptions = BitmapFactory.Options().apply {
                // Get the dimensions of the bitmap
                inJustDecodeBounds = true
                Log.d(TAG,
                    "Current Photo path is$currentPhotoPath Width:$outWidth Height:$outHeight"
                )

                BitmapFactory.decodeFile(currentPhotoPath, this)
                val photoW: Int = outWidth
                val photoH: Int = outHeight

                // Determine how much to scale down the image
                var scaleFactor = 1
                if(photoH>0 && photoW>0) {
                    scaleFactor = max(1, min(photoW / targetW, photoH / targetH))
                }
                // Decode the image file into a Bitmap sized to fill the View
                inJustDecodeBounds = false
                inSampleSize = scaleFactor
                inMutable = true
            }
            val exifInterface = ExifInterface(currentPhotoPath)
            val orientation = exifInterface.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
            )

            val bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
            return when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> {
                    rotateImage(bitmap, 90f)
                }
                ExifInterface.ORIENTATION_ROTATE_180 -> {
                    rotateImage(bitmap, 180f)
                }
                ExifInterface.ORIENTATION_ROTATE_270 -> {
                    rotateImage(bitmap, 270f)
                }
                else -> {
                    bitmap
                }
            }
        }catch (e: Exception){
            Log.e(TAG,
                "Error in get captured image...the current photo path is...$currentPhotoPath",e)
            throw e
        }
    }

    /**
     * getSampleImage():
     *      Get image form drawable and convert to bitmap.
     */
    private fun getSampleImage(drawable: Int): Bitmap {
        Log.d(TAG, "Getting sample image.")

        return BitmapFactory.decodeResource(resources, drawable, BitmapFactory.Options().apply {
            inMutable = true
        })
    }

    /**
     * rotateImage():
     *     Decodes and crops the captured image from camera.
     */
    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        Log.d(TAG, "Rotating image.")
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    /**
     * createImageFile():
     *     Generates a temporary image file for the Camera app to write to.
     */
    @Throws(IOException::class)
    private fun createImageFile(): File {
        Log.d(TAG, "creating image file.")
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    /**
     * dispatchTakePictureIntent():
     *     Start the Camera app to take a photo.
     */
    private fun dispatchTakePictureIntent() {
        Log.d(TAG, "Taking picture.")
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    Log.d(TAG, "Saving captured image.")
                    createImageFile()
                } catch (e: IOException) {
                    Log.e(TAG, e.message.toString(), e)
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    Log.d(TAG, "File has been saved. Continuing.")
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "org.tensorflow.codelabs.objectdetection.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    /**
     * drawDetectionResult(bitmap: Bitmap, detectionResults: List<DetectionResult>
     *      Draw a box around each objects and show the object's name.
     */
    private fun drawDetectionResult(
        bitmap: Bitmap,
        detectionResults: List<DetectionResult>
    ): Bitmap {
        Log.d(TAG, "Drawing detection result.")
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val rPen = Paint()
        rPen.color = Color.RED
        rPen.textAlign = Paint.Align.LEFT
        rPen.strokeWidth = 8F
        rPen.style = Paint.Style.STROKE

        val yPen = Paint()
        yPen.style = Paint.Style.FILL_AND_STROKE
        yPen.color = Color.YELLOW
        yPen.strokeWidth = 2F
        yPen.textSize = MAX_FONT_SIZE


        detectionResults.forEach {
            // draw bounding box
            val box = it.boundingBox
            canvas.drawRect(box, rPen)

            val tagSize = Rect(0, 0, 0, 0)

            // calculate the right font size
           // var lengthOfText = 0
//            if(it.text != null) {
//               val lengthOfText = it.text.length
//            }
            yPen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = yPen.textSize * box.width() / tagSize.width()

            // adjust the font size so texts are inside the bounding box
            if (fontSize < yPen.textSize) yPen.textSize = fontSize

            var margin = (box.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F
            canvas.drawText(
                it.text, box.left + margin,
                box.top + tagSize.height().times(1F), yPen
            )
        }
        return outputBitmap
    }

}

/**
 * DetectionResult
 *      A class to store the visualization info of a detected object.
 */
data class DetectionResult(val boundingBox: RectF, val text: String)
