package org.tensorflow.codelabs.objectdetection

class OcrParameters {
    companion object{
        const val maxResults: Int = 10
        const val scoreThreshold = 0.2f

    }
}

class DetParameters {
    companion object{
        const val maxResults: Int = 10
        const val scoreThreshold = 0.4f
    }
}