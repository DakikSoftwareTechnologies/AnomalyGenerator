# Anomaly Generation

The amount of accessible data, coming from real-life situations, that can be used to evaluate and validate anomaly detection models is really low and this makes the process of comparing and choosing the best methods very use case dependent. For example, a use case may have a very tight definition of what an anomaly is and the approaches yielding the best results for such a use case may be useless for another use case with a more loose definition of an anomaly. To prevent wasting time and resources to find and select the best approaches from a wide variety of them, knowing how these approaches perform in general and in comparison to each other would help tighten the search pool and lower the amount of effort needed to obtain good results for any use case. To achieve such a comparison method one would need to have access to labeled anomalous data on demand and since it has been established that this is not realistic, another approach is to generate anomalies using clean data.

## Random Noise

From a data mining perspective, the word "noise" is used to describe any part of a dataset that is usually undesired, which may or may not be outliers/anomalies. These sections of data are undesired because they do not contain any meaningful information about the overall sample. One of the most common types of noise used is called Gaussian noise. Gaussian noise has a mean of zero and a standard deviation of one, meaning it has the same probability distribution function as the normal distribution and can easily be generated for any type of dataset. It is used in many ways such as making neural network training processes more robust [[1]](#1), [[2]](#2), training autoencoders suited for de-noising [[3]](#3), and anomaly generation for model evaluation and testing [[4]](#4).

## Generative Adversarial Networks and Variational Autoencoders

Generative Adversarial Networks (GANs) are a class of deep learning algorithms focused on learning the rules and generalities within a dataset to then be able to generate new examples that look natural within the context of the original dataset. The success shown by GANs has led to researchers using them for anomaly generation. In [[5]](#5), the authors use GANs on a one-dimensional host-based intrusion dataset to generate anomalies. This is done by setting clean data as one of the distributions and anomalous data as the other to teach the network how to transform normal data into anomalous data. This approach may be successful, but it is limited by the need to acquire labeled anomaly data before the generation.

Variational Autoencoders (VAEs) also show similar success when it comes to generating data from a latent representation. In [[6]](#6), a VAE is trained to learn distributions using statistical information from the original data, and then these distributions are sampled to obtain the latent representation. If at one point an anomaly is going to be introduced, what the VAE does is a sample from the outliers of the distributions learned for that point, so when the decoder uses the latent representation to reconstruct the data, the generated anomalies will be coming from these points. This approach is also shown to be successful and there is no need for previously acquired anomaly data to generate new anomalies.

## Model Evaluation and Validation

The main reason anomaly generation is needed and why it is a research area getting more and more interesting, and the main reason why this project exists is because of the need to evaluate and validate implemented models. After using methods detailed in previous sections and many others to obtain the necessary training and testing data, evaluation and validation become easier for any use case. To compare and evaluate models, metrics from [[7]](#7) have been implemented but will not be detailed here.

## How to Use

### 1) Running the script directly:

```cli
usage: test.py [-h] -t TASK -f INPUT -d OUTDIR

Generate anomalous data or evaluate anomaly detection output

optional arguments:
 -h, --help                  show this help message and exit
 -t TASK, --task TASK        Task to perform, "ano" or "eval"
 -f INPUT, --input INPUT     Input file
 -d OUTDIR, --outdir OUTDIR  Directory for output
```

There are two modes, anomaly generation and evaluation. If you run the
script with the “-t” flag set to “ano”, anomaly generation will be performed
according to the input file provided.The input file specifications for anomaly
generation are as follows:

```json
{
  "train_params": {
    "columns": ["Temperature", "Humidity"],
    "sequence_length": 10,
    "latent_space_dim": 8,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 150,
    "path": "../path/to/training/data",
    "key": "date",
    "dates": [["2015-02-10 00:00:00", "2015-02-10 09:33:00"]],
    "nrows": 8144
  },
  "noise_params": {
    "anomally_normal_ratio": 0.005,
    "noise": 0.1,
    "wave": false,
    "wave_length": 10,
    "columns": ["Temperature", "Humidity"],
    "noise_direction": "p",
    "path": "..path/to/training/data",
    "csv_mode": "single",
    "key": "date",
    "dates": [["2015-02-08 05:09:00", "2015-02-10 09:33:00"]],
    "nrows": 8144
  },
  "imdir": "../vae_img3"
}
```

Most of the parameters are related to the training process of the neural
network that will be used to generate anomalies. “columns” is for the
features that will be used, if the array is left empty all of the features will be
used. “path” is the location of the data file. “dates” are used to split data
into training and testing sets, data from between these two dates will be
used as testing data (this will be the set that will have generated
anomalies) and the rest will be used to train the network. If you add more
than one pair here, there will be an array of train and testing sets with corresponding indices in the array (train index 0 -> test index 0). “imdir” is
the place where the output images of both the anomalous data and the
clean data will be stored so users can have visual context. If the string is
left empty no image will be generated. At the end of the execution, a csv
file that contains anomalous data and a json file indicating the indices of
the anomalous points in the data will be stored in the directory provided
with the “-d” flag.
If you run the script with the “-t” flag set to eval, anomaly detection
evaluation will be performed. An example input file for this process would
be as follows:

```json
{
  "targets": [
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0]
  ],
  "candidates": [
    [1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0]
  ],
  "range": 3
}
```

### 2) Importing functions to a custom project:

It is possible to import the two main functionalities and helpers they use to
a custom python project, this way you can customize how the pipeline
works according to your needs. You can prepare python objects in your
project to be used as input to the main processes of anomaly generation
and evaluation as described above. The functions only return the
objects without saving them into files so you can use them as-is. An API Reference is being created to help with this.
There is also a function that generates a regular LSTM autoencoder which can be trained with the same inputs as the variational autoencoder. It is left
there so if there is a need to use an additional anomaly detection model
and reach it quickly the method can be used to obtain an anomaly
detection model within the pipeline or the custom project.
Hyperparameters for both the variational autoencoder and the regular
autoencoder are hardcoded, you might need to play with them to get
better results for a specific set of data. Hyperparameter tuning addition is one of the next steps for this project.

### Examples:

|                                                     Clean Data                                                      |                                                   Anomalous Data                                                   |
| :-----------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| <img src ="https://github.com/DakikSoftwareTechnologies/AnomalyGenerator/blob/main/images/1randomNoiseClean.png" width="600" height= "360"> | <img src="https://github.com/DakikSoftwareTechnologies/AnomalyGenerator/blob/main/images/1randomNoise.png" width="600" height= "360"> |
| <img src="https://github.com/DakikSoftwareTechnologies/AnomalyGenerator/blob/main/images/0randomNoiseClean.png" width="600" height= "360"> | <img src="https://github.com/DakikSoftwareTechnologies/AnomalyGenerator/blob/main/images/1randomNoiseClean.png" width="600" height= "360" > |

## References

<a id="1">[1]</a>
Guozhong An. “The Effects of Adding Noise During
Backpropagation Training on a Generalization Performance”. In: Neural Computation 8.3 (1996), pp. 643–674.
URL: https://doi.org/10.1162/neco.1996.8.3.643

<a id="2">[2]</a>
Arvind Neelakantan et al. Adding Gradient Noise Improves Learning for Very Deep Networks. 2015. URL: https://arxiv.org/abs/1511.06807.

<a id="3">[3]</a>
Pascal Vincent et al. “Stacked Denoising Autoencoders:
Learning Useful Representations in a Deep Network
with a Local Denoising Criterion”. In: Journal of
Machine Learning Research 11.110 (2010), pp. 3371– 3408.
URL: http://jmlr.org/papers/v11/vincent10a.html.

<a id="4">[4]</a>
Stelios C. A. Thomopoulos and Christos Kyriakopoulos. “Anomaly detection with noisy and missing data
using a deep learning architecture”. In: Signal Processing, Sensor/Information Fusion, and Target Recognition
XXX. Ed. by Lynne L. Grewe, Erik P. Blasch, and
Ivan Kadar. SPIE, Apr. 2021. URL: https://doi.org/10.1117/12.2589981.

<a id="5">[5]</a>
Milad Salem, Shayan Taheri, and Jiann Shiun Yuan.
“Anomaly Generation Using Generative Adversarial
Networks in Host-Based Intrusion Detection”. In: 2018
9th IEEE Annual Ubiquitous Computing, Electronics
Mobile Communication Conference (UEMCON). 2018,
pp. 683–687. DOI: 10 . 1109 / UEMCON . 2018 . 8796769.
URL: https://arxiv.org/abs/1812.04697

<a id="6">[6]</a>
Nikolay Pavlovich Laptev. “AnoGen : Deep Anomaly
Generator Nikolay Laptev”. In: 2018.

<a id="7">[7]</a>
Kovács, György & Sebestyen, Gheorghe & Hangan, Anca. (2019). Evaluation
metrics for anomaly detection algorithms in time-series. Acta Universitatis
Sapientiae, Informatica. 11. 113-130.
URL: https://doi.org/10.2478/ausi-2019-0008
