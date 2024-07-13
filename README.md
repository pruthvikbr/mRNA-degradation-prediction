# mRNA-degradation-prediction
Degradation rates of mRNA sequence bases based on their bpps, type of loop structures formed.

**Sumary**: Implemented Bi-directional LSTM and Bi-direction GRU chained together to form a neural network to predict the following five output  variables.Implemented feature engineering for eacch molecule calculatin gbase pair probability, their pairing stdandard deviation. Converting Character values into numerical categorical values . The model is run with 60 epochs which took a time of 1 and half days for me. The performance of the model is calculated using MCRMSE( mean column wise root mean square) . where rmse is calculated for all the predicted columns with the test set and it's average is taken. For private testset, the model scored mcrmse of 37.39% but for public testset it acheived 25.57% which performed better than attempted previously.Right now it's one among the best but need to improve a bit more on this to be the number one. 

For each molecule id in the test set, we must predict targets for each sequence position. If the length of the sequence of an id is, e.g., 107, then you should make 107 predictions. Positions greater than the seq_scored


Variables predicted: reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C

Definition of variables:  

**deg_Mg_50C**:An array of floating point numbers, should have the same length as seq_scored. These numbers are reactivity values for the first 68 bases as denoted in sequence, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).

**deg_50C** - An array of floating point numbers, should have the same length as seq_scored. These numbers are reactivity values for the first 68 bases as denoted in sequence, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).

**deg_Mg_pH10** - An array of floating point numbers, should have the same length as seq_scored. These numbers are reactivity values for the first 68 bases as denoted in sequence, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10)

**deg_pH10** - An array of floating point numbers, should have the same length as seq_scored. These numbers are reactivity values for the first 68 bases as denoted in sequence, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10)

**reactivity** - An array of floating point numbers, should have the same length as seq_scored. These numbers are reactivity values for the first 68 bases as denoted in sequence, and used to determine the likely secondary structure of the RNA sample.

**Credit: Tito for exolaining me about the project and the initial approach,Google and dive into deep learning book**
