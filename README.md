# OpenUIE-Benchmark
An open-source benchmark for underwater image enhancement.

OpenUIE-Benchmark is an open-source project designed to make underwater image enhancement methods easier to reproduce, evaluate, and compare.

The project aims to provide a unified and beginner-friendly framework for running classical and deep learning-based underwater image enhancement methods, using publicly available pretrained models, and evaluating their performance under consistent protocols.

## 1. Project Goals

Underwater image enhancement methods have developed rapidly. However, reproducing different methods often requires different dependencies, model implementations, preprocessing pipelines, and evaluation scripts.

OpenUIE-Benchmark aims to address these challenges by providing:

- **Easy reproduction:** Run enhancement methods through beginner-friendly Google Colab notebooks.
- **Unified evaluation:** Evaluate different methods using consistent image processing and metric implementations.
- **Fair comparison:** Compare enhancement quality and computational efficiency under clearly documented protocols.
- **Open and reproducible research:** Record model sources, pretrained weights, configurations, and evaluation settings.

## 2. Planned Methods

The project will gradually integrate classical and deep learning-based methods.

| Category | Method | Status |
|---|---|---|
| Traditional | UDCP | Planned |
| Traditional | Fusion-based Enhancement | Planned |
| Traditional | CLAHE | Planned |
| Deep Learning | WaterNet | Planned |
| Deep Learning | UWCNN | Planned |
| Deep Learning | FUnIE-GAN | Planned |
| Deep Learning | UIEC²-Net | Planned |
| Deep Learning | NU2Net | Planned |

Methods will be included in the runnable benchmark after their source code, dependencies, pretrained weights, and usage permissions have been checked.

## 3. Evaluation Metrics

The benchmark is designed to cover three aspects of performance.

### 3.1 Full-Reference Image Quality Metrics

These metrics require paired reference images:

- PSNR
- SSIM
- MS-SSIM
- LPIPS
- FSIM
- PCQI

### 3.2 No-Reference Image Quality Metrics

These metrics do not require reference images:

- UIQM
- UCIQE
- MUSIQ
- URanker

### 3.3 Computational Efficiency

- Number of Parameters
- FLOPs / MACs
- Inference Time
- Frames Per Second (FPS)
- Peak GPU Memory
- Model Size

Not every metric applies to every method. For example, traditional methods without trainable parameters will be marked as N/A for parameter count.

Full-reference metrics will only be calculated when valid paired reference images are available. Reference images are not necessarily physical ground truth.

## 4. Supported Datasets

Potential datasets for future integration include:

- UIEB
- EUVP
- LSUI

Dataset availability, download sources, directory conventions, and reference-image definitions will be documented separately.

Users should download datasets from their original sources and store them in their own Google Drive.

## 5. Quick Start

The project is designed to run on Google Colab.

The intended workflow is:

1. Open a notebook from this GitHub repository in Google Colab.
2. Mount your own Google Drive.
3. Select or upload underwater images.
4. Select an enhancement method.
5. Run enhancement and save the results.
6. Calculate image quality and computational metrics.
7. Export comparison figures and CSV reports.

**Note:** The first runnable release is under development. Instructions will be updated as individual methods become available.

## 6. Project Structure

```text
OpenUIE-Benchmark/
├── notebooks/       # Google Colab notebooks
├── src/             # Core implementation
├── configs/         # Model and evaluation configurations
├── docs/            # Documentation
└── examples/        # Example images and results
```

## 7. Reproducibility

To support reproducible experiments, the benchmark will record:

- Python and deep learning framework versions
- GPU model and runtime environment
- Model implementation and checkpoint sources
- Preprocessing and postprocessing configurations
- Evaluation metric implementations and settings
- Inference time and computational resource usage

Pretrained weights and datasets will not be redistributed unless their licenses explicitly permit redistribution. Users should follow the original authors' licenses and terms.

## 8. License

The original code developed for this project is released under the MIT License.

Third-party implementations, pretrained weights, and datasets remain subject to their respective licenses and usage conditions.

## 9. Contributions

Contributions are welcome.

Researchers and developers are encouraged to contribute:

- Reproducible implementations of underwater image enhancement methods
- Verified pretrained model integrations
- Evaluation metric implementations
- Dataset documentation
- Bug reports and improvements

Please preserve original copyright notices and provide references to the original publications and implementations.

## 10. Project Status

**Current stage: Initial development**

The first release will focus on:

1. UDCP implementation
2. WaterNet integration
3. Unified image quality evaluation
4. Google Colab one-click workflow
5. Basic multi-method comparison

More methods and evaluation capabilities will be added progressively.

---

**Repository:** [OpenUIE-Benchmark](https://github.com/lxgcoding/OpenUIE-Benchmark)

## Development Environment

- Code Editor: Visual Studio Code
- Execution Platform: Google Colab
- GPU: NVIDIA Tesla T4
- Version Control: Git and GitHub