# Setup to run ONNX models in QuPath on Windows 11

To run ONNX models in QuPath, we need to add the ONNX Runtime engine to Deep Java Library (DJL). 
If you have an NVIDIA GPU, according to the [DJL docs](https://djl.ai/engines/onnxruntime/onnxruntime-engine/), it is possible to run ONNX Runtime via CUDA.

We will need to change a couple of things in the **QuPath** source code to enable a build that includes the propper JARs.  

I have an **NVIDIA GeForce RTX 4070 Ti SUPER**, and it worked for me. I’m not sure whether it works with the RTX 50 series.  

Let’s go step by step.

## Download QuPath source code and do a couple of changes

Following the [QuPath docs](https://qupath.readthedocs.io/en/stable/docs/reference/building.html), to build from source you need **OpenJDK 21 or higher** and **Git**. I assume you already have them installed.

In my folder called `qupath-source`, I run the following in `cmd`:

```
git clone https://github.com/qupath/qupath
cd qupath
```
Then open QuPath folder in VSCode, since we need to make a couple of changes.

You need to find the file: qupath.djl-conventions.gradle.kts

It is located here: `\qupath_source\qupath\buildSrc\src\main\kotlin\qupath.djl-conventions.gradle.kts`

Once you have this file, you can do two things:

First, you can download the file with the same name from this repository and replace it.

Alternatively, find the following lines in the file (they appear twice):

```
	if ("onnx" in djlEngines || "onnxruntime" in djlEngines) {
	    implementation("ai.djl.onnxruntime:onnxruntime-engine:$djlVersion")
	    // No model zoo available
	}
```

And replace it by:  

```
if ("onnx" in djlEngines || "onnxruntime" in djlEngines) {

		val useOrtGpu = providers.gradleProperty("djl.onnxruntime.gpu")
			.getOrElse("false")
			.toBoolean()

		val ortVersion = providers.gradleProperty("djl.onnxruntime.version")
			.getOrElse("1.21.1")

		if (useOrtGpu) {
			implementation("ai.djl.onnxruntime:onnxruntime-engine:$djlVersion") {
				exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime")
			}
			runtimeOnly("com.microsoft.onnxruntime:onnxruntime_gpu:$ortVersion")
		} else {
			implementation("ai.djl.onnxruntime:onnxruntime-engine:$djlVersion")
		}

		

		// No model zoo available
	}
```


> [!IMPORTANT]  
> Crreate gradle.properties file


Now you need to create (or download, then paste) the gradle.properties file.

This tells QuPath to download the ONNX Runtime (with GPU support) and PyTorch engines. The contents are:

```
org.gradle.parallel=true
djl.engines=pytorch,onnxruntime
djl.onnxruntime.gpu=true
djl.onnxruntime.version=1.21.1
```

Save the changes and build from source (the first build may take a few minutes) using this command:

```
gradlew clean jpackage
```
I have the QuPath build in this directory:

`C:\Users\Usuario\Desktop\qupath_source\qupath\build\dist\QuPath-0.7.0-SNAPSHOT`

However, you still need to create a launch script to enable GPU support.


## Create a Conda environment and a launch script for this build

Both engines need CUDA to run, but finding the correct version can be tricky.

So I’m sharing a `qupath-ort-pytorch-gpu.txt` file to set up the Conda environment. To create the environment, run the following in the Anaconda Prompt:


```
conda create --name qupath-ort-pytorch-gpu --file qupath-ort-pytorch-gpu.txt
```

Then it’s time to create a launch script (a .bat file). Interestingly, thanks to ChatGPT, this is a really useful option:

I created `qupath-ort-pytorch-gpu.bat` in VS Code, and it contains the following:


```
@echo off
set CONDA_PREFIX=C:\Users\Usuario\anaconda3\envs\qupath-ort-pytorch-gpu

set PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\DLLs;%CONDA_PREFIX%;%PATH%

"C:\Users\Usuario\Desktop\qupath_source\qupath\build\dist\QuPath-0.7.0-SNAPSHOT\QuPath-0.7.0-SNAPSHOT (console).exe"
```

Run `qupath-ort-pytorch-gpu.bat` and **QuPath-0.7.0-SNAPSHOT** will open.

Then go to `Extensions > DeepJavaLibrary > Manage DJL engines`.

Interestingly, at least on my PC, **DJL** automatically downloads **PyTorch 2.7.1** with GPU support. You will also get the **ONNX Runtime** engine.


<img width="420" height="479" alt="image" src="https://github.com/user-attachments/assets/a6623dc0-95ad-4f36-ab7f-3efd8f4a4e62" />


You can check whether CUDA is detected by opening the Script Editor and running:

```
println ai.djl.engine.Engine.getEngine("OnnxRuntime")
println ai.djl.engine.Engine.getEngine("PyTorch")
```

I have this output: 
```
INFO: OnnxRuntime:1.21.1, OnnxRuntime:1.21.1, capabilities: [
	MKL,
	CUDA]
INFO: PyTorch:2.7.1, capabilities: [
	CUDA,
	CUDNN,
	OPENMP,
	MKL,
	MKLDNN,
]
PyTorch Library: C:\Users\Usuario\.djl.ai\pytorch\2.7.1-cu128-win-x86_64
```

# Install the StarDist extension for QuPath

Go to `Extensions > Manage Extensions`, find **QuPath StarDist extension v0.6.0**, install it, and restart QuPath using `qupath-ort-pytorch-gpu.bat`.

If you downloaded the StarDist models converted to **ONNX** with the Colab Notebook, you can use the scripts in the `inference_StarDist_Onnx` folder.
There is one inference script for `he_heavy_augment.onnx` and one for `dsb2018_heavy_augment.onnx`.

These scripts use **QuPath StarDist extension v0.6.0**, but for inference they use DJL with ONNX Runtime.

> [!NOTE]  
> This is not a pure ONNX pipeline. It uses DJL’s [Hybrid Engine](https://docs.djl.ai/master/docs/hybrid_engine.html) with PyTorch. In short, the translator runs with the PyTorch engine, which is why you need both engines installed.


I can’t guarantee it works everywhere, but I hope it does! :)

I’ll make a video tutorial soon.











