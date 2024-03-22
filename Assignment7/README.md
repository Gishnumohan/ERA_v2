## Assignment - 7
### In Depth coding practice 

The targets for this assignment are as follows: consistently achieving a validation accuracy of 99.4% in the last few epochs, completing training in less than or equal to 15 epochs, and ensuring that the model has fewer than 8000 parameters. Achieve these goals using modular code, with each model being included in the model.py file as Model_1, Model_2, etc

### Model_1
Target:
        -Set Basic Working Code
        -Set Transforms
        -Set Data Loader  
        -Basic training & test loop
Results:
        Parameters: 10,988
        Best Training Accuracy: 98.96%
        Best Test Accuracy: 98.91%
Analysis:
        Heavy Model for such a problem

### Model_2

Target:
    Add Regularization, Dropout
    Increase model capacity. Add more layers at the end.
Results:
    Parameters: 7,400
    Best Training Accuracy: 98.80%
    Best Test Accuracy: 99.16%
Analysis:
    Able to reduce the model size less than 8000 parameters


### Model_3

Target:
    Add LR Scheduler
Results:
    Parameters: 7,276
    Best Training Accuracy: 98.60%
    Best Test Accuracy: 99.24%
Analysis:
    Need to acheive 99.4% accuracy

