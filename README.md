# TPR
## Técnicas de Percepção de Rede (Network Awareness)

This repository contains the code for setting up and running a server-client architecture designed for this RAT project.

The server is hosted on a Virtual Machine (VM) running Ubuntu, and clients are managed through a remote access trojan (RAT).

## Getting Started

These instructions will guide you through the setup process for the server and client.

### Prerequisites

- A VM with Ubuntu (any recent version should work)
- Python 3.x installed on both the server and client machines

### Server Setup

1. **Create a VM with Ubuntu**:
   - Set up a Virtual Machine running Ubuntu. You can use platforms like VirtualBox, VMWare, or any cloud-based VM providers like AWS, GCP, or Azure.

2. **Secure Copy of `server.py`**:
   - Use SCP or a similar tool to securely copy `src/server.py` to your VM.

3. **Install Requirements**:
   - Install all required Python packages. The required packages are listed in the imports of `server.py`.
   - You can usually install these packages using pip. For example:
     ```bash
     pip3 install [package-name]
     ```

4. **Certificate Setup**:
   - In your VM, set up a certificate for secure communication.
     ```bash
     openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
     ```
   - When prompted, set a password for the certificate. For the country, use `PT` for Portugal, and for other fields, you can input `.` if you prefer not to specify.

### Client Setup
  1. **Install RAT requirements**:
   - Install all required Python packages. The required packages are listed in the imports of `server.py`.
   - You can usually install these packages using pip. For example:
     ```bash
     pip3 install [package-name]
     ```
     
## Usage
### Run the server:
   - Inside the VM, run `python3 server.py`
     
### Running the RAT:
   - On the client machine, run `rat.sh`. This script will execute `sudo python3 rat.py`.
   - Ensure that `rat.py` is in an accessible location on the client machine.

## Sampling Process

The project includes a Python script for sampling network traffic data. This script (`basePktSampling.py`), which is inside the directory scripts/processing processes packet data and extracts relevant information based on specified client and service networks.

The data is then sampled at a defined interval to create a manageable and representative dataset.

### Key Features of the Sampling Script:

- **Filtering by Network**: The script allows specifying client (`-c` or `--cnet`) and service (`-s` or `--snet`) network prefixes. Only traffic involving these networks is considered for sampling.
  
- **Custom Sampling Interval**: You can set the sampling interval (`-d` or `--delta`) to control the granularity of the data. This interval defines how often data is sampled and aggregated.

- **Support for Multiple File Formats**: The script can handle different formats of network data, including script format, Tshark format, and pcap format.

- **Output File Generation**: The script generates an output file (`-o` or `--output`) with the sampled data, including statistics like packet counts and average packet sizes.

### Running the Sampling Script:

To execute the sampling script, use the following command structure:

```bash
python3 basePktSampling.py -i [input file] -o [output file] -f [format] -d [delta] -c [client network(s)] -s [service network(s)]
```

## Profiling Process (Feature Extraction)

The profiling process involves extracting features from network traffic. This is achieved through `basePktFeaturesSilenceExt.py`. 

### Script Overview: `basePktFeaturesSilenceExt.py`

The `basePktFeaturesSilenceExt.py` script is designed for advanced feature extraction from network traffic data. The key functions and their roles are as follows:

- `extractStats(data)`: Extracts basic statistical features such as mean, median, and standard deviation from the traffic data.

- `extractStatsAdv(data, ignoreSilence, threshold)`: Similar to `extractStats`, but with the capability to ignore periods of silence (inactivity) in the traffic.

- `extratctSilenceActivity(data, threshold)`: Segregates the traffic data into 'silence' and 'activity' based on a defined threshold.

- `slidingObsWindow(data, lengthObsWindow, slidingValue)`: Utilizes a sliding window approach to extract features over specified intervals, providing a more dynamic analysis of the traffic.

### Usage

To use the script, run it with the necessary arguments specifying the input file, observation window method, window widths, and sliding value:

```bash
python basePktFeaturesSilenceExt.py -i <input_file> -m <method> -w <window_widths> -s <slide_value>
```

## Classification

The `baseClassification.py` script is responsible for performing classification tasks on network traffic data. It includes several key features:

- **Feature Loading**: Loads pre-processed feature files, including data for various classes like 'Files', 'Browsing', 'Images', 'Streaming'.
- **Data Preparation**: Splits the data into training and testing sets with a specified percentage.
- **Machine Learning Models**:
  - **K-Means Clustering**: Used for clustering data into groups.
  - **DBSCAN Clustering**: Another clustering method to separate high density areas from low-density regions.
  - **Support Vector Machines (SVM)**: Three different kernels (Linear, RBF, and Polynomial) are used for classification tasks.
  - **Neural Networks**: Utilizes Multi-Layer Perceptron (MLP) for classification.
- **Performance Metrics Calculation**: After classification, calculates various performance metrics like precision, true positives, and false positives.

### Key Functions in Script

1. **Clustering with K-Means and DBSCAN**: The script performs clustering to understand the data distribution and find patterns within the data.

2. **Classification with Support Vector Machines (SVM)**: Uses different kernels to classify the data into predefined categories.

3. **Classification with Neural Networks**: Employs an MLP classifier for predicting the classes of the network traffic data.

### Usage

To use the `baseClassification.py` script, ensure that all the dependencies are installed as per the requirements. Load your pre-processed feature files and specify the desired configuration for the classifiers.

```bash
python baseClassification.py
```

## Anomaly Detection: Real and Mixed Traffic

The `baseAnomalyDetectionRealMixed.py` script focuses on detecting anomalies in network traffic, particularly distinguishing between normal and RAT (Remote Access Trojan) activities. It incorporates various methods and visualization techniques for anomaly detection.

### Key Features and Methods

- **Feature Loading**: Loads the pre-processed feature files for normal and mixed traffic data.
- **Data Splitting**: Divides the data into training and testing sets based on specified percentages.
- **Anomaly Detection Techniques**:
  - **One-Class Support Vector Machines (SVM)**: Uses different kernels (Linear, RBF, and Polynomial) to detect anomalies.
  - **Gaussian Mixture Models (GMM)**: Employs Gaussian Mixture Models for identifying outliers in data.
  - **Isolation Forest (IF)**: Utilizes the Isolation Forest algorithm for anomaly detection.
- **Performance Metrics**: Calculates true positives, false positives, true negatives, and false negatives. Also computes precision, recall, F1 score, and accuracy.
- **Visualization**: Includes functions to plot confusion matrices for a visual representation of model performance.

### Usage

To utilize the `baseAnomalyDetectionRealMixed.py` script:

1. Load the feature datasets for normal and mixed traffic.
2. Configure the percentage split for training and testing datasets.
3. Run anomaly detection models and observe the output.
4. Analyze the results using the provided metrics and confusion matrix plots.

```bash
python baseAnomalyDetectionRealMixed.py
```
