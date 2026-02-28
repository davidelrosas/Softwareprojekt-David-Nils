# Final project by Nils Enno Lenwerder & David Daniel Rosas Marquez

The idea for our final project was to test wether adding a mechanism of attention to a model would truly improve it's performance on the PhysioNet sepsis classification dataset. With this goal in mind we combined both of our works to make a homogeneous environment to test the models against each other fairly.

## DESCRIPTION

The experimental setup consisted of training three different models with the PhysioNet Sepsis Data and then comparing their performance on the classification task. The Models we used for this project were a GRU-ODE as a baseline ("BASE_model.py"), ACE-GRU-ODE ("ACE_model.py" where the only difference to the ACE-ODE-RNN described in the Readme of the folder "DAVID", is that this one uses a GRU cell from equinox instead of a standard RNN cell) and a different Attention based model we made ("ATT_model.py"). Each model was trained on 80% of the dataset which was processed with the same preprocessing pipeline each time. The performance of the models was then validated on the remaining 20% of the dataset, by using F1 and recall scores and the confusion matrix of the predictions. 

The experiment (training the model, validating on the dataset and saving the metrics) was performed 10 times for each model. At the end we used non parametric statistical tests (Kruskal-Wallis Test and Pairwise Mann-Whitney U Test) to check wether there was a significant difference on the performance between models.

To recreate the experiment, the PhysioNet sepsis dataset (which is not contained inside of this folder) is needed. First the "makefile" must be run, to create the training and validation data from the patient time series files. Then either of the 3 models can be imported into the "experiment.py" script. Running this script will train and validate the model, and save the metrics from the run in an output file named "training_history.npz". These files need to be placed inside a folder, that must be provided to the "statsig.py" script before running it. This file then compares the performance of the models with each other.

## MODELS

### BASE_MODEL

This is a Recurrent Neural Network model with a GRU connection instead of a standard RNN.

### ACE_MODEL

This model was heavily based on the architecture presented by Jhin et al., 2021, using a set of two co-evolving ODEs to evolve both the hidden state and the attention matrix. This version mainly differs from the one presented in the paper in the training pipeline. We decided that training both ODEs separately (by freezing one and then the other), was not worth the extra computational cost.

### ATT_MODEL

This is a model with a static learned attention that gets applied to the hidden state on each step of the integration, but that doesn't co-evolve with it.


## RESULTS

When Comparing the three models based on their performance on the recall metric, there was no significant difference. This means all three models were equally as good at reducing the amount of false negatives. When running the statistical tests based on the F1 metric, there was no significant difference when comparing the ACE-Model and the Base-Model, and when comparing the ATT-Model and the Base-Model with each other. There was, however, a difference when comparing the ACE-Model and the ATT-model with each other. In general, the ACE-Model slightly outperformed the ATT-Model in its ability to keep false positives low. 


## REFLECTION

All in all our results showed us, that at least with our current experimental setup and potentially unoptimized models, the addition of a mechanism of attention to the models didn't increase their accuracy for this classification task. Still, when performing an "eye test" the ACE-model does seem to yield overall slightly better results than it's counterpart without the attention. That's why it would be interesting to explore wether the results we got during this project are due to a poor optimization of the ACE-model or if this architecture of co-evolving attention and hidden state is just not suited for this specific task and could fare better when tested on other problem sets.

## Sources

Jhin, S. Y., Jo, M., Kong, T., Jeon, J., & Park, N. (2021). *ACE-NODE: Attentive Co-Evolving Neural Ordinary Differential Equations*. arXiv preprint arXiv:2105.14953. https://arxiv.org/abs/2105.
