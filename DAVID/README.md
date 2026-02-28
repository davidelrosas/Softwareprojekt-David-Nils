# Softwareproject by David Daniel Rosas Marquez

Hello and welcome to my part of the Softwareproject. In the Following Text I'll give a brief overview of my work throughout the semester, that I did for the course "Softwareprojekt: Machinelles Learnen für Lebenswissenschaftliche Daten". But first a couple of notes:

---

## Notes

1) This folder doesn't contain all of the scripts I wrote throughout the semester, just the ones I thought are the most relevant to the course and the "final attempts" for each project. I'll also always specify the files that need to be run and in which order, and if specific folder structures are needed to be able to reproduce the intended output.

2) A lot of the python scripts are extensively commented (especially for the first couple projcets). I chose to leave the comments in, because they were part of my process to try and better understand certain machine learning concepts (As I was completely new to the topic), and to rationalize what was happening in the code. But this also means that a lot of the comments might contain unclear or outright wrong statements, a lot of typos, grammar errors, redundancy and so on. This goes especially for comments where I am trying to understand/explain certain concepts.

3) The naming conventions for classes and variables and the coding style can be somewhat inconsistent. The code structure for the projects is also somewhat messy and not necessarily optimaly modular. I hope this doesn't cause too much confusion.

4) I divided my work into Projects called "Project_x" that thematically fit together. Some of them took multiple attempts or multiple weeks, but I decided to only upload the final working versions. I did this to provide an easier to follow overview of my work thoughout this semester.

---

With this in mind now follows the Overview.

---

# PROJECT 0

The First couple of weeks for me mostly involved getting into the topic of machine learning and artificial neural networks. For this we first implemented a simple linear regression using gradient descent, a technique also used for learning in neural networks. In the script "linear_regression.py" (runs on it's own) the idea is to interpolate some data points that follow a simple linear function with noise. Some diagrams for the loss surface are also plotted, to better be able to visualize the effects of gradient descent. 
The other project is in the script "digit_recog.py" (runs on it's own), which contains code for a simple multilayer perceptron trained to recognize digits from the mnist_784 dataset. This project got me to learn the basics of machine learning and artificial neural networks.
During the first couple of weeks we also had to do some reading on more complex NODE models, for which I chose the ACE_NODE. In "ACE_NODE.odp" you can see the slides for the presentation I made on the topic.

**difficulties:**  
Trying to build the sufficient understanding to begin tackling the basic concepts of Artificial Neural Networks and machine learning. Also trying to understand the paper "ACE-NODE: Attentive Co-Evolving Neural Ordinary Differential Equations" as it was full of concepts that were completely new to me.

**learned:**  
-What is a neural network
-Forward Pass and Backpropagation (superficially)  
-how do neural networks "learn" using gradient descent  
-How to code and train a basic neural network
-Basics of classification tasks
-Basics of Jax

---

# PROJECT 1

By this point I had already developed a basic understanding of some concepts in machine learning and ANNs. The next project consisted in starting our journey through the world of Neural Ordinary Differential Equations, a machine learning paradigm consisting of modeling the continous dynamics of a systems state. To this end I coded minimal version of a NODE in the script "minimal_node.py" (runs on it's own) and tested it's interpolation capabilities on a synthetic dataset of datapoints following a sinus curve. The idea here was to implement it without the use of dedicated libraries (other than jax) and to create our own numerical integration function using Euler's method. This attempt was sadly not very succesful and the model fails to interpolate the sinus curve, even without noise (probably due to some bug). The next week our task was to implement the same minimal architecture but with the use of the dedicated libraries equinox (for neural network), diffrax (for the integrator) and optax (for the optimizer). We tested the model on a dataset of covid infections over time for interpolation, and on a dataset of lynx and hare populations over time for interpolation and extrapolation. I also tested the model on the same synthetic dataset as before but with vastly better results. All in all the greatest accomplishment of my model was it's ability to capture the periodic behaviour of the populations for some runs of the extrapolation task. It also performed okay for the interpolation of covid cases. This can be tested out with the "min_node.py" script by choosing the experiment in the main function. It requires the "covid_data.npy" and "LH_data.npy" files to run.

**difficulties:**
The difficulties for this task were mainly to understand the differences of this new ML paradigm to the previous discrete model I had coded to recognize digits. Getting familiar with the Python libraries was also a big part of the work during these weeks. Another difficulty I had was to feed the node model with the lynx and hare population data. Up to this point it was not intuitive to me that time series could consist of multiple features. So thanks to the lynx and hare dataset I became more familiar with the concept of multivariate time series and how to feed the model with this data correctly, so that it could learn the interactions between the different variables.

**learned:**
-How to implement a minimal Node Architecture and a simple integrator using Euler's method
-The basics of the equinox, diffrax and optax libraries
-The importance of Normalizing the data and tweaking the Hyperparameters of a model to achieve better results
-What a multivariate time series is (through lynx and hare experiment)

---

# PROJECT 2

The next project involved trying to predict a latent variable alpha for a dataset of spiral trajectories. For this we couldn't keep using the simple NODE architecture but had to implement a model caple of learning latent variables with more ease. I ended up implementing an ODE RNN model, a recurrent neural network that also uses a NODE to evolve it's hidden state. The first week of this project our dataset consisted of spiral trajectories containing 100 uniformely sampled datapoints. More interesting was the following week, where we tested one of the most important advantages of the NODE architecture, which is to handle irregularly sampled data. For this we had to train the model to predict the latent variable alpha on now four different datasets of spiral trajectories, and compare the performance on each dataset. The first again with 100 uniformely sampled points and then subsequently with 75, 50 and 25 irregularly sampled datapoints per trajectory. The results showed that this implementation of the model is able to retain a pretty good degree of accuracy even when the sampling frequency is reduced. This can be tested by running the script "environment.py" which needs all other files inside the folder to run properly. 

**difficulties:**
Because we had to handle a more complex dataset and model, I had to properly learn to utilize the concepts of vectorizing and scanning in the context of jax. This was challenging at first but drastically improved the runtime of the application. The vectorization let us handle the batches of this very large dataset more efficiently, while the scanning was a crucial part of implementing the RNN in a way that was compilable for jax.

**learned:**
-About ANNs with Recurrent connections
-To use the hidden state as a latent representation of the dynamics of the system
-The advantages of NODE architecture for sparsely and irregularly sampled data
-Different types of normalization methods
-Vmap and scan

---

# PROJECT 3

From this point on all the projects were done in preparation for our final project (This one can be found in the Final_Project Folder of the repo). The idea was to implement a model from litarature, for which we chose the ACE-NODE, and test it on a classification task on the Physionet Sepsis dataset. The first couple of weeks I had to get familiar with the concepts of attention and coevolving ODEs presented inside the paper "ACE-NODE: Attentive Co-Evolving Neural Ordinary Differential Equations". To test these out I implemented the model under "ACE_NODEv3.py". It consists of 2 coevolving ODEs, one for the hidden state and another for the attention matrix. This attention matrix is applied to the hidden state as a pairwise attention on every step of the integration, as described by the paper. By feeding both ODEs with this new vector, not only is the evolution of the hidden state modulated by the attention matrix, but the attention evolves dynamically according to the changes in the hidden state. The paper argues, that this "attentive co-evolving NODE concept is also
an effective way to increase the representation learning capability of NODEs" (Jhin et al., 2021, p. 3). Also interesting is that both ODEs are trained separately with different loss function, one after the other, by updating only the weights of the ODE that is currently being trained. To test this model I used the Lynx and Hare population dataset again for interpolation and extrapolation. The results for this first attempt at implementing the model were rather bad. Although it did managed to capture the periodicity of the populations on the extrapolation task, this happened muche mor infrequently than with the simple NODE architecture. But this may be due to the attention based model not being suited for this specific task or because of poor optimization. To recreate the experiment run the script "test_model.py".

**difficulties:**
There were mainly two difficulties with this project. First was to implement the training cycle proposed by the authors of the paper. This required for me to learn how to freeze specific parts of the model in equinox, so that the two ODEs could be trained separately. But still with the model properly working, it was really difficult to get it to behave correctly for both the interpolation and the extrapolation tasks. Ultimately I didn't manage to get it to have a consistent performance.

**learned:**
-About Pairwise attention, coevolution and other concepts needed to implement an ACE NODE
-How to integrate systems consisting of more than one ODE with the diffrax solver
-How to work with the PyTree model representation in Equinox

---

# PROJECT 4

The paper by Jhin et al., 2021 we took the model from, conveniently contains a section for a very similar experiment as our final project. Here they compare the performance of different models on the PhysioNet Mortality classification dataset. Specifically they wanted to test wether their attention based model, ACE-Latent-ODE (ODE Enc.), would perform better than the others. This gave me the idea to combine the ACE mechanism with the ODE RNN architecture I had previously learned. The ACE Node would be used to evolve the hidden state to the point right before the observation arrives, and an RNN cell would then combine the state with the new observation for the next step. This decision made sense to me for the PhysioNet Sepsis classification task, because the dataset seemed to require a model capable of handling irregularly sampled and sparse data, multivariate time series of vastly different lengths, and latent dynamics. This led to the creation of the model in the script "ACE_ODE_RNNv2.py", which was tested on the datasets of spiral trajectories already presented in Project 2 (run the script "environment.py" to recreate the experiment). Unfortunately as in the previous project, the results were somewhat underwhelming. Although the model managed to predict the alpha variable for the 4 different datasets succesfully, it seemed to performe slightly worse than the ODE RNN. And to top it off, the computing time required to train the ACE ODE RNN is much greater than that of the ODE RNN. This happens because the attention vector that needs to be applied to the hidden state at each step of the integration and that is outputed by the attention ODE, has the dimension of the hidden state squared. Also the fact that both the hidden state and the attention ODEs have to be trained separately increases this runtime.

**difficulties:**

The biggest difficulties this project were to decide how to generate the initial attention and how to forward the attention to the next step on the recurrent connection. The paper roughly described how the initial attention was generated for each experiment, but didn't meantion how to handle the attention on recurrent connections. To gain some clarity on this issue I wrote an email to the corresponding author of the paper, but sadly I have yet to recieve a response. For the time being on this project I decided to generate the initial attention with an outer product and to reset the attention with the same iniitial attention generation function on each step of the RNN.

**learned:**
-To combine aspects of different models into a new functioning (but not necessarily good) model

---

# PROJECT 5

With the main framework of the ACE ODE RNN completed, it was finally time to modify it to be able to process the PhysioNet Sepsis dataset. For this I first had to create a preprocessing pipeline to handle the 20000 patient time series files. The idea here was to get rid of empty files, and organize the data in a more usable way for the model. To this end I created a new dictionary file containing for each patient: 

- The Dynamic hourly vital signs and lab measurements
- The static demographic information
- A mask for the time series to encode when a value was observed (1) or was not observed (0)
- The label denoting wether the patient got sepsis after 6 hours (1) or not (0)

The normalization statistics were then computed fot the whole dataset. The model was modified accordingly, to handle the dynamic data part of the time series as the observations, while the static data for the patients was injected into the initial hidden state of each time series. This way the model is supposed to learn to evolve the hidden state taking the demographic information from the patient into account. The initial attention was generated by creating a correlation matrix of the initial hidden state for all patient files in the batch. This means all time series would share the same initial attention, but this matrix would then continue to evolve with each respective hidden state independently from that point on. The way the recurrent connection was handled was also changed from the previous Project. Instead of generating a new initial attention for each step of the RNN, the attention computed by the ACE NODE would be passed through it's own RNN cell. When training the model all time series were truncated or padded to a specific length before being fed to the model, so that the batches could be used for the vectorized forward pass. 

After testing the performance of the model on the PhysioNet Sepsis classification task, the results seemed to be quite promising, but to be sure I modified an ODE RNN to handle the dataset in the same way, and compared the performance of both models. Unfortunately it was quite difficult to tell wether the Attention mechanism provided any substantial benefit over the much faster to train ODE RNN. This was a question that I tred to answer in my Final project, working together with my group partner Nils.

To recreate the experiment, the scrtipt "preprocessing.py" has to be run on the PhysioNet sepsis dataset first (which is not contained inside the fodler) to create the files the model is to be trained with. After this the desired model to be tested out (ACE ODE RNN or ODE RNN) should be imported into "environment.py" before runing the script.

**difficulties:**
Although learning to make a model able to handle such a complex dataset was difficult in it's own right. What I struggled the most with this time was to design the code structure in an organized and modular way. With so many moving parts the handling of the preprocessing ended up being split between the "preprocessing.py" and "training.py" scripts in a way that I was not entirely happy with.


**learned:**
- Various techniques for data preprocessing such as masking and padding
- To Apply various concepts of machine learning to succesfully train a model with a complex dataset
- That a bigger and more complex model is not necessarily always better
- The importance of benchmarking, which was further explored in the final project

---

# REFLECTION

(The last project overview can be found under the folder ""FINAL_PROJECT" in this repo.)

Throughout this whole semester I got familiar with multiple concepts of Machine Learning, spanning from different NODE based architectures to different model training and preprocessing techinques. I was also able to explore these topics and apply my learnings to my own models with varying degrees of success. It was really interesting to observe the ability of NODEs to handle irregularly sampled data and learn the dynamics of the systems on real and synthetic datasets. In total I managed to recreate my own version for 2 NODE architectures from literature, one being the ODE RNN and the other the ACE NODE. Taking inspiration from the paper by Jhin et al., 2021 I also applied the concept of attention and coevolving ODEs to the ODE RNN to create my own ACE ODE RNN. By succesfully applying this model to the PhysioNet Sepsis classification task, I was able to deepen my understanding of machine learning, Artificial Neural Networks and NODEs. 

Throughout this course we mainly worked with supervised learning algorithms. In the future I am looking to explore ANNs in contexts that require unsupervised and reinforced learning algorithms. Im also excited to learn more about architectures that use Transformers, Convolutional Neural Networks or ones that try to mimick biologically plausible systems, like Spiking Neural Networks.



## Sources

Jhin, S. Y., Jo, M., Kong, T., Jeon, J., & Park, N. (2021). *ACE-NODE: Attentive Co-Evolving Neural Ordinary Differential Equations*. arXiv preprint arXiv:2105.14953. https://arxiv.org/abs/2105.14953