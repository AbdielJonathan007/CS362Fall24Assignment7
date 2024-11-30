### Assignment 7: Neural models. 

#### Due May 16, 11:59pm. 

*note: No late assignments are allowed for this assignment.*

*note also - there's no starter code for this assignment. Just this README.*

**Task 1: Perceptrons. (30 points)** 

[(from Russell and Norvig, 18.27)](https://aimacode.github.io/aima-exercises/concept-learning-exercises/ex_27/)

Consider the following set of training examples A1-A14, each with six inputs X1-X6 and one
target output:

<table>
<tr><td>Example</td><td>A1</td><td>A2</td> <td>A3</td> <td>A4</td> <td>A5</td> <td>A6</td> <td>A7</td> <td>A8</td> <td>A9</td> <td>A10</td> <td>A11</td> <td>A12</td> <td>A13</td> <td>A14</td>      </tr>
<tr><td> X1  </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 0 </td></tr>
<tr><td> X2 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 1 </td></tr>
<tr> <td>	X3  </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td></tr>
<tr><td>	X4  </td><td> 0 </td><td> 1  </td><td> 0  </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 0 </td><td> 1 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 1 </td></tr>
<tr><td>	X5  </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 0 </td></tr>
<tr><td>	X6  </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 0 </td></tr>
<tr><td>	Output   </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 0 </td></tr>
</table>

Write a python function to implement the perceptron learning algorithm on this data and show the final weights. 
Please include this in a file called perceptron.py. Assume that alpha = 0.1, and include a bias unit=-0.1.
Since the data is linearly separable, your code should be able to find a solution that correctly classifies all of the data.

**Task 2: TensorFlow.(40 points)**

TensorFlow is a well-known and widely used deep learning package. Google has provided a large set of tutorials and resources to help people learn to use it.
Most of them work in CoLab, so you should not need to install anything locally if you don't want to.

I've identified three tutorials that I think are helpful in getting you started with TensorFlow, and also drawing some comparisons to other techniques we've learned about.

These tutorials are great, but it's easy to wind up just clicking on things and not really absorbing the material. In order to help you think about 
some of the interesting issues being presented, I've developed a set of questions for you to answer. Please add a document with the answers to your repo. 

NOTE: Please answer these questions in your own words - do not just cut and paste from the tutorial. Cut-and-paste answers (or ChatGPT-generated answers) will receive zero credit.

First, please work through these tutorials on Convolutional Networks: 

https://www.tensorflow.org/tutorials/images/cnn

https://www.tensorflow.org/tutorials/images/classification

- What is a MaxPooling2D layer? What's it do?

- What's Adam? 

- What's the softmax function do? 

- What is CategoricalCrossEntropy? What do we use it for?

- In the CNN example, what does the Flatten layer do?

- In the CNN example, what does the Dense layer do? 

- In the CNN example, why does the height and width get smaller for each convolutional layer?

Next, please work through this tutorial on preprocessing:
https://www.tensorflow.org/tutorials/load_data/csv

- What does it mean to normalize the data? Where else have we seen normalization? 

- Why is it a problem that the Titanic data has different types and ranges? Why did we not have to worry about this with the decision tree?

- What is a one-hot vector?

- The example that shows how to manually slice the feature dictionary uses yield instead of return.
Why is this? What's the difference between them, and why would you want to use yield?

Finally, please work through this tutorial on recurrent NNs: https://www.tensorflow.org/text/tutorials/text_classification_rnn

- As we know, encoding is a particularly important part of working with neural networks. Explain how text is encoded for an RNN. 

- What does the Bidirectional layer do? What are the advantages and disadvantages of this approach? How does it compare to the way we processed sequence data with an HMM?

- What is masking? Why do we need to use it in this example?

**Task 3: Applying Neural Models. (30 points)**

- [This tutorial](https://huggingface.co/learn/deep-rl-course/unit3/introduction?fw=pt) shows how to use the HuggingFace platform and the RLZoo framework to implement Deep Q-learning to solve Atari 2600 games.
  (note: it takes about 9 hours to train this network if you train for 10M steps. I recommend reducing this significantly, at least the first few times you run.)

1. Work through it. (you might want to look back to Chapter 2 in the tutorial if you've forgotten Q-learning.)
2. Solve another game that's not Space Invaders. 
3. Publish your model to the Hub and email me a link to it.
4. Include the colab code in your repo.

**Grad students:**

- [This tutorial](https://huggingface.co/docs/transformers/tasks/question_answering) shows how to use HuggingFace transformers to do question answering, a core NLP task.
1. Work through it.
2. Write a wrapper for your question answerer to allow users to chat with it. Include this in your repo. (you can include it in the colab notebook if you want.)
3. Publish model to the Hub and email me a link to it.
4. Include the colab code in your repo.
