/*
 * For macOS users
 * 
 * Script to run StarDist models converted to ONNX using the StarDist extension for QuPath.
 *
 * This script is for `he_heavy_augment.onnx`.
 *
 * The normal QuPath build-from-source setup with ONNX Runtime should be enough for macOS.
 *
 * ONNX Runtime does not support MPS acceleration yet: https://github.com/microsoft/onnxruntime/issues/21271
 * 
 * However, ONNX Runtime in DJL should be able to support CoreML: https://github.com/deepjavalibrary/djl/blob/master/engines/onnxruntime/onnxruntime-engine/src/main/java/ai/djl/onnxruntime/engine/OrtModel.java#L225
 * 
 * If the StarDist operations are supported by CoreML, it should work. You can try onnx models with fixed 512 inputs if this one does not work.
 * 
 * See the NOTE on line 120.
 * 
 * GitHub repo: https://github.com/iviecomarti/stardist_qupath_onnx_pytorch
 */

// -------------------------
// User params
// -------------------------
def onnxPath = "path/to/he_heavy_augment.onnx"


// -------------------------
// Run StarDist2D using ONNX via DJL
// -------------------------
def dnn = new OnnxDjlDnnModel(onnxPath, device)

// Use builder(DnnModel) to bypass QuPath’s DjlDnnModelBuilder. 
def stardist = StarDist2D.builder(dnn)
        .normalizePercentiles(1, 99) // Percentile normalization
        .tileSize(512)
        .threshold(0.2)              // Probability (detection) threshold
        .pixelSize(0.5)              // Resolution for detection
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add nucleus measurements
        .doLog()
        .build()



// Run on selected annotation(s)
def imageData = getCurrentImageData()
def parents = getSelectedObjects()
if (parents == null || parents.isEmpty()) {
    print "Select one or more parent annotations to run StarDist.\n"
} else {
    stardist.detectObjects(imageData, parents)
    println "StarDist done."
}


// Cleanup
try { stardist.close() } catch (ignored) {}
try { dnn.close() } catch (ignored) {}






import static qupath.lib.gui.scripting.QPEx.*

import qupath.ext.stardist.StarDist2D
import qupath.opencv.dnn.DnnModel
import org.bytedeco.opencv.opencv_core.Mat

import ai.djl.Device
import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.repository.zoo.Criteria
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import ai.djl.translate.Batchifier
import ai.djl.training.util.ProgressBar

import qupath.ext.djl.DjlTools



// -------------------------
//  DnnModel adapter that runs ONNX via DJL
// Generated with help of ChatGPT
// -------------------------
class OnnxDjlDnnModel implements DnnModel, AutoCloseable {

    private final String modelPath
    private final Device device
    private final Model model
    private final Translator<NDArray, NDArray> translator
    private final ThreadLocal predictorTL

    OnnxDjlDnnModel(String modelPath, Device device) {
        this.modelPath = modelPath
        this.device = device

        // Translator
        this.translator = new Translator<NDArray, NDArray>() {
            @Override
            NDList processInput(TranslatorContext ctx, NDArray input) {
                return new NDList(input)
            }

            @Override
            NDArray processOutput(TranslatorContext ctx, NDList list) {
                return list.singletonOrThrow()
            }

            @Override
            Batchifier getBatchifier() {
                return null
            }
        }
        
        //Criteria 
        //NOTE: If StarDist operations are supported by CoreML, it might use the GPU
        //key change at line 129, setting "CoreML". optDevice maybe is not be needed
        //Source: https://djl.ai/engines/onnxruntime/onnxruntime-engine/
        def criteria = Criteria.builder()
                .setTypes(NDArray.class, NDArray.class)
                .optModelPath(java.nio.file.Paths.get(modelPath))
                .optTranslator(translator)
                .optEngine("OnnxRuntime")
                //.optDevice(device)
                .optOption("ortDevice", "CoreML")
                .optProgress(new ProgressBar())
                .build()

        this.model = criteria.loadModel()

        // Predictor per thread
        this.predictorTL = ThreadLocal.withInitial({
            return this.model.newPredictor(this.translator)
        })
    }

    Map<String, Mat> predict(Map<String, Mat> inputs) {

        Mat matIn = inputs.values().iterator().next()
        def predictor = predictorTL.get()

        // Use ONNXRuntime manager
        def manager = model.getNDManager().newSubManager()
        try {
            int h = matIn.rows()
            int w = matIn.cols()
            int c = matIn.channels()

            // Mat -> NDArray in HWC
            NDArray xHwc = DjlTools.matToNDArray(manager, matIn, "HWC")
                    .toType(ai.djl.ndarray.types.DataType.FLOAT32, false)

            // If no normalization is done in the extension, you can use this instead
            //xHwc = xHwc.div(255f)

            // Build NHWC 
            float[] xArr = xHwc.toFloatArray()
            NDArray x = manager.create(xArr, new ai.djl.ndarray.types.Shape(1, h, w, c))

            // Inference: expect [1,H,W,33]
            NDArray y = predictor.predict(x)

            // Avoid squeeze(); recreate [H,W,33]
            float[] yArr = y.toFloatArray()
            NDArray yHwc = manager.create(yArr, new ai.djl.ndarray.types.Shape(h, w, 33))

            // NDArray -> Mat, to use it by Stardist Extension
            Mat matOut = DjlTools.ndArrayToMat(yHwc, "HWC")
            return ["output": matOut]
        } finally {
            manager.close()
        }
    }

    @Override
    void close() {
        // Close predictors
        try {
            def p = predictorTL.get()
            if (p != null) p.close()
        } catch (ignored) {}

        // Close model
        if (model != null) model.close()
    }
}

