# Setup to run ONNX models in QuPath on macOS (best guess)

Running ONNX models on macOS should be easier than on Windows 11.

Unfortunately, I don’t have access to an **Apple Silicon Mac** right now, but I did some tests while developing **DANEELpath**.  I managed to run **DANEELpath** U-Nets with ONNX on macOS, and  GPU was being used  I *think* via CoreML (I think…)  
However, I don’t know whether the **StarDist** operations will work. I’m sharing the setup here just in case.

## Build QuPath from source

You just need to follow the [QuPath docs](https://qupath.readthedocs.io/en/stable/docs/reference/building.html) and build from source. Basically, you just need to add the `gradle.properties` file (or edit it) with:

```
org.gradle.parallel=true
djl.engines=pytorch,onnxruntime
```

Then build QuPath using:

```
./gradlew clean jpackage 
```


You should not need any kind of launch script here. You just need to open  **QuPath-0.7.0-SNAPSHOT** and add the engine here: 
`Extensions > DeepJavaLibrary > Manage DJL engines`

However, if you run this script in QuPath:
```
println ai.djl.engine.Engine.getEngine("OnnxRuntime")
```

You will see no MPS and no CoreML. Actually, MPS is not supported yet in ONNX Runtime: https://github.com/microsoft/onnxruntime/issues/21271

However,  ONNX Runtime in **DJL** should be able to support CoreML: https://github.com/deepjavalibrary/djl/blob/master/engines/onnxruntime/onnxruntime-engine/src/main/java/ai/djl/onnxruntime/engine/OrtModel.java#L225

As I said, the U-Nets from **DANEELpath** used the GPU from a macOS M1. However I do not know if it will work for **Stardist** 

You will find in the folder `inference_stardist_onnx_macos` the scripts with the criteria builder adapted for CoreML. 








